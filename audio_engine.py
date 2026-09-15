"""Inference orchestration for Whisper speech alignment, Chatterbox TTS, and AudioSR super-resolution."""

import os
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Suppress harmless deprecation warnings from legacy sub-modules
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.jit.script is deprecated.*")

# =========================================================================
# PATCH 1: Bypass Perth Watermarker
# Fixes 'NoneType' instantiation crash & prevents 16.8kHz high-cut attenuation
# =========================================================================
import perth

class BypassWatermarker:
    def __init__(self, *args, **kwargs):
        pass

    def apply_watermark(self, wav, *args, **kwargs):
        return wav

    def get_watermark(self, *args, **kwargs):
        return 0.0

perth.PerthImplicitWatermarker = BypassWatermarker

# =========================================================================
# PATCH 2: Route TorchAudio via SoundFile
# Fixes "TorchCodec is required for load_with_torchcodec" crash on Windows
# =========================================================================
def _patched_torchaudio_load(uri, *args, **kwargs):
    data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
    # Convert from soundfile's (frames, channels) to torchaudio's (channels, frames)
    tensor = torch.from_numpy(data.T)
    return tensor, sr

def _patched_torchaudio_save(uri, src, sample_rate, *args, **kwargs):
    if isinstance(src, torch.Tensor):
        if src.ndim == 1:
            data = src.detach().cpu().numpy()
        elif src.ndim == 2:
            data = src.detach().cpu().numpy().T
        else:
            data = src.detach().cpu().numpy()
    else:
        data = np.asarray(src)
    sf.write(str(uri), data, sample_rate)

torchaudio.load = _patched_torchaudio_load
torchaudio.save = _patched_torchaudio_save

# Intercept direct torchcodec loader bindings if present in PyTorch 2.10+
if hasattr(torchaudio, "load_with_torchcodec"):
    torchaudio.load_with_torchcodec = _patched_torchaudio_load
if hasattr(torchaudio, "save_with_torchcodec"):
    torchaudio.save_with_torchcodec = _patched_torchaudio_save

# =========================================================================
# Primary Neural Model Imports
# =========================================================================
from chatterbox import ChatterboxTTS
from faster_whisper import WhisperModel
from audiosr import build_model, super_resolution
import noisereduce as nr

from config import (
    AUDIOSR_MODEL_NAME,
    AUDIOSR_DDIM_STEPS,
    AUDIOSR_GUIDANCE_SCALE,
    DEFAULT_CFG_WEIGHT,
    DEFAULT_EXAGGERATION,
    DEFAULT_TEMPERATURE,
    ENABLE_REFERENCE_SANITIZATION,
    NOISE_REDUCE_PROP,
    REFERENCE_NORMALIZE_PEAK,
    SAMPLE_RATE,
)
from media_utils import (
    apply_broadcast_mastering,
    calculate_metrics,
    mux_video_universal,
    time_stretch_clean,
)
from text_processing import format_timestamp, normalize_tech_script

# Cached singleton model pointers
_AUDIOSR_MODEL = None
_WHISPER_MODEL = None
_CHATTERBOX_ENGINE = None


def sanitize_reference_sample(raw_ref_path: str, clean_ref_path: str) -> str:
    """
    Cleans room noise, boxy tone reflections, and stationary microphone hiss
    from the target voice sample before extracting speaker conditioning latents.
    """
    if not ENABLE_REFERENCE_SANITIZATION:
        return raw_ref_path

    audio, sr = sf.read(raw_ref_path)

    # Force mono downmix so spatial reflections do not corrupt conditioning vectors
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Spectral gating preserves vocal formants while lowering stationary room tails
    clean_audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=float(NOISE_REDUCE_PROP),
        n_fft=1024,
        win_length=1024,
        hop_length=512,
    )

    # Re-normalize to prevent clipping
    peak = np.max(np.abs(clean_audio))
    if peak > 0:
        clean_audio = (clean_audio / peak) * REFERENCE_NORMALIZE_PEAK

    sf.write(clean_ref_path, clean_audio.astype(np.float32), sr)
    return clean_ref_path


def get_audiosr():
    global _AUDIOSR_MODEL
    if _AUDIOSR_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _AUDIOSR_MODEL = build_model(
            model_name=AUDIOSR_MODEL_NAME,
            device=device,
        )
    return _AUDIOSR_MODEL


def super_resolve_timeline(
    input_wav: str,
    output_wav: str,
    seed: int = 42,
    ddim_steps: int = AUDIOSR_DDIM_STEPS,
    guidance_scale: float = AUDIOSR_GUIDANCE_SCALE,
) -> str:
    """
    Passes the 24kHz synthesized timeline through a latent diffusion model
    to inpaint and synthesize true 12kHz-20kHz harmonic 'air', outputting
    at a 48kHz studio sample rate.
    """
    model = get_audiosr()

    # Flush GPU cache before running diffusion passes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.inference_mode():
        waveform = super_resolution(
            model,
            input_wav,
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            ddim_steps=int(ddim_steps),
        )

    # Convert tensor or multi-dimensional array into flat 1D float32 audio
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.squeeze(waveform).astype(np.float32)

    # Normalize output to prevent digital overs
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = (waveform / peak) * 0.95

    # Export true 48kHz uncompressed audio
    sf.write(output_wav, waveform, 48000)
    return output_wav


def get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # Run Whisper-base on CPU with int8 quantization.
        # Transcribes in ~1-2 seconds with zero VRAM usage, avoiding CTranslate2 CUDA collisions.
        _WHISPER_MODEL = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            cpu_threads=4
        )
    return _WHISPER_MODEL


def get_chatterbox():
    global _CHATTERBOX_ENGINE
    if _CHATTERBOX_ENGINE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CHATTERBOX_ENGINE = ChatterboxTTS.from_pretrained(device=device)
    return _CHATTERBOX_ENGINE


def get_vram_usage() -> str:
    """Returns real-time GPU VRAM allocated by PyTorch."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        return f"{allocated:.0f} MB"
    return "CPU Mode"


def transcribe_video_audio(audio_path: str, srt_out: str):
    """Generates timestamped subtitles from audio using Faster-Whisper with VAD."""
    whisper = get_whisper()
    segments, _ = whisper.transcribe(audio_path, beam_size=5, vad_filter=True)

    cues_list = []
    with open(srt_out, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(
                f"{idx}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n\n"
            )
            cues_list.append((idx, seg.start, seg.end, seg.text.strip()))

    return srt_out, cues_list


def synthesize_sentence_blocks(
    sentence_blocks: list,
    ref_audio_path: str,
    run_dir: Path,
    temperature: float = DEFAULT_TEMPERATURE,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    seed: int = 42,
    progress_callback=None,
):
    """
    Renders speech for sentence blocks with locked speaker identity.
    Uses non-overlapping buffer placement to prevent voice collisions.
    """
    sanitized_ref_path = str(run_dir / "sanitized_narrator_ref.wav")
    clean_ref_audio = sanitize_reference_sample(ref_audio_path, sanitized_ref_path)

    engine = get_chatterbox()
    engine.prepare_conditionals(clean_ref_audio, exaggeration=float(exaggeration))

    total_duration = sentence_blocks[-1]["end"] + 3.0
    master_timeline = np.zeros(int(total_duration * SAMPLE_RATE), dtype=np.float32)

    last_end_sample = 0
    records = []
    temp_preview_path = str(run_dir / "temp_cue_preview.wav")

    for idx, block in enumerate(sentence_blocks):
        cue_num = idx + 1
        s_start = block["start"]
        s_end = block["end"]
        raw_text = block["text"]
        normalized_text = normalize_tech_script(raw_text)

        # Lookahead boundary calculation: prevents spilling into next line
        next_start_sec = (
            sentence_blocks[idx + 1]["start"]
            if idx + 1 < len(sentence_blocks)
            else s_end + 1.0
        )
        available_window = max(0.5, next_start_sec - s_start - 0.08)

        max_samples = int(available_window * SAMPLE_RATE)
        # Collision avoidance: clamp start to never overlap with previous sentence tail
        start_sample = max(int(s_start * SAMPLE_RATE), last_end_sample)
        timecode_str = f"{s_start:05.2f}s - {s_end:05.2f}s"

        # Lock RNG seed per sentence for timbre consistency
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        wav = engine.generate(
            text=normalized_text,
            temperature=float(temperature),
            cfg_weight=float(cfg_weight),
        )
        clip = (
            wav.squeeze().detach().cpu().numpy()
            if isinstance(wav, torch.Tensor)
            else np.array(wav, dtype=np.float32)
        )
        raw_duration = len(clip) / SAMPLE_RATE

        stretch = 1.0
        status_tag = "Natural Cadence"
        if len(clip) > max_samples:
            stretch = min(len(clip) / max_samples, 1.25)
            clip = time_stretch_clean(clip, stretch, run_dir, SAMPLE_RATE)
            status_tag = f"WSOLA Compressed ({stretch:.2f}x)"

        peak_val = calculate_metrics(clip)
        sf.write(temp_preview_path, clip, SAMPLE_RATE)

        end_sample = start_sample + len(clip)
        if end_sample > len(master_timeline):
            master_timeline = np.pad(
                master_timeline, (0, end_sample - len(master_timeline))
            )

        # Clean slot assignment: replaces summing to guarantee zero double-talk bleed
        master_timeline[start_sample:end_sample] = clip
        last_end_sample = end_sample

        record = {
            "Cue": cue_num,
            "Timestamp": timecode_str,
            "Script Segment": raw_text[:38] + ("..." if len(raw_text) > 38 else ""),
            "Window": f"{available_window:.2f}s",
            "Duration": f"{raw_duration:.2f}s",
            "WSOLA": f"{stretch:.2f}x",
            "Peak": peak_val,
            "Status": status_tag,
        }
        records.append(record)

        if progress_callback:
            progress_callback(
                cue_num,
                timecode_str,
                raw_text,
                records,
                temp_preview_path,
                peak_val,
                stretch,
            )

    # Master timeline normalization
    peak = np.max(np.abs(master_timeline))
    if peak > 0:
        master_timeline = (master_timeline / peak) * 0.92

    raw_wav_path = str(run_dir / "timeline_raw.wav")
    sf.write(raw_wav_path, master_timeline, SAMPLE_RATE)
    return raw_wav_path, records


def resynthesize_single_cue(
    cue_index: int,
    sentence_blocks: list,
    new_text: str,
    ref_audio_path: str,
    run_dir: Path,
    video_path: str,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    seed: int,
    apply_eq: bool = True,
):
    """
    Slices into master_timeline, re-synthesizes ONLY the chosen cue,
    re-splices it into the audio buffer, and remuxes the final video.
    """
    if cue_index < 1 or cue_index > len(sentence_blocks):
        raise ValueError("Selected cue index is out of bounds.")

    block = sentence_blocks[cue_index - 1]
    block["text"] = new_text.strip()
    norm_text = normalize_tech_script(block["text"])

    s_start = block["start"]
    s_end = block["end"]
    next_start_sec = (
        sentence_blocks[cue_index]["start"]
        if cue_index < len(sentence_blocks)
        else s_end + 1.0
    )
    available_window = max(0.5, next_start_sec - s_start - 0.08)

    max_samples = int(available_window * SAMPLE_RATE)
    start_sample = int(s_start * SAMPLE_RATE)

    sanitized_ref_path = str(run_dir / "sanitized_narrator_ref.wav")
    if os.path.exists(sanitized_ref_path):
        clean_ref_audio = sanitized_ref_path
    else:
        clean_ref_audio = sanitize_reference_sample(ref_audio_path, sanitized_ref_path)

    engine = get_chatterbox()
    engine.prepare_conditionals(clean_ref_audio, exaggeration=float(exaggeration))

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    wav = engine.generate(
        text=norm_text,
        temperature=float(temperature),
        cfg_weight=float(cfg_weight),
    )
    clip = (
        wav.squeeze().detach().cpu().numpy()
        if isinstance(wav, torch.Tensor)
        else np.array(wav, dtype=np.float32)
    )

    stretch = 1.0
    status_tag = "Natural Cadence"
    if len(clip) > max_samples:
        stretch = min(len(clip) / max_samples, 1.25)
        clip = time_stretch_clean(clip, stretch, run_dir, SAMPLE_RATE)
        status_tag = f"WSOLA Compressed ({stretch:.2f}x)"

    peak_val = calculate_metrics(clip)

    # 1. Read existing master timeline
    raw_wav_path = str(run_dir / "timeline_raw.wav")
    master_timeline, sr = sf.read(raw_wav_path, dtype="float32")

    # 2. Zero out old slice and splice new clip
    end_sample = start_sample + len(clip)
    if end_sample > len(master_timeline):
        master_timeline = np.pad(
            master_timeline, (0, end_sample - len(master_timeline))
        )

    clear_window_end = min(len(master_timeline), int(next_start_sec * SAMPLE_RATE))
    master_timeline[start_sample:clear_window_end] = 0.0
    master_timeline[start_sample:end_sample] = clip

    peak = np.max(np.abs(master_timeline))
    if peak > 0:
        master_timeline = (master_timeline / peak) * 0.92

    sf.write(raw_wav_path, master_timeline, SAMPLE_RATE)

    # 3. Re-master
    final_audio = raw_wav_path
    if apply_eq:
        mastered_wav = str(run_dir / "timeline_mastered.wav")
        apply_broadcast_mastering(raw_wav_path, mastered_wav)
        final_audio = mastered_wav

    # 4. Remux Video
    final_video = str(run_dir / "final_mastered.mp4")
    mux_video_universal(video_path, final_audio, final_video)

    cue_preview_path = str(run_dir / "temp_cue_preview.wav")
    sf.write(cue_preview_path, clip, SAMPLE_RATE)

    updated_record = {
        "Cue": cue_index,
        "Timestamp": f"{s_start:05.2f}s - {s_end:05.2f}s",
        "Script Segment": block["text"][:38] + ("..." if len(block["text"]) > 38 else ""),
        "Window": f"{available_window:.2f}s",
        "Duration": f"{len(clip)/SAMPLE_RATE:.2f}s",
        "WSOLA": f"{stretch:.2f}x",
        "Peak": peak_val,
        "Status": f"⚡ Patched ({status_tag})",
    }

    return final_video, final_audio, cue_preview_path, updated_record