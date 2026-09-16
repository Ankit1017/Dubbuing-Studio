"""Inference orchestration for Whisper speech alignment, Chatterbox TTS, and AudioSR super-resolution."""

import os
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


def is_valid_file(file_path: Optional[str | Path], min_bytes: int = 1024) -> bool:
    """Verifies whether an on-disk artifact exists and has a valid file size."""
    if not file_path:
        return False
    p = Path(file_path)
    return p.exists() and p.is_file() and p.stat().st_size >= min_bytes


def sanitize_reference_sample(raw_ref_path: str, clean_ref_path: str) -> str:
    """Cleans room noise, boxy tone reflections, and stationary hiss before latent extraction."""
    if not ENABLE_REFERENCE_SANITIZATION:
        return raw_ref_path

    if is_valid_file(clean_ref_path):
        return clean_ref_path

    audio, sr = sf.read(raw_ref_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    clean_audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=float(NOISE_REDUCE_PROP),
        n_fft=1024,
        win_length=1024,
        hop_length=512,
    )

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


def get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # Offloaded to CPU (int8) to guarantee zero VRAM collision with PyTorch/Demucs
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


def transcribe_video_audio(audio_path: str, srt_out: str) -> Tuple[str, List[Tuple[int, float, float, str]]]:
    """Generates timestamped subtitles using Faster-Whisper with VAD."""
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


def super_resolve_timeline(
    input_wav: str,
    output_wav: str,
    seed: int = 42,
    ddim_steps: int = AUDIOSR_DDIM_STEPS,
    guidance_scale: float = AUDIOSR_GUIDANCE_SCALE,
    chunk_duration: float = 5.12,
    overlap_duration: float = 0.5,
    telemetry_callback: Optional[Callable[[int, int, float, str], None]] = None
) -> str:
    """
    Passes audio through AudioSR in 5.12s windows with linear crossfading.
    Provides live telemetry feedback for each diffusion chunk.
    """
    if is_valid_file(output_wav):
        if telemetry_callback:
            telemetry_callback(1, 1, 100.0, "Cached 48kHz audio restored from disk.")
        return output_wav

    model = get_audiosr()
    audio_data, in_sr = sf.read(input_wav, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    total_samples = len(audio_data)
    chunk_samples = int(chunk_duration * in_sr)
    overlap_samples = int(overlap_duration * in_sr)
    step_samples = chunk_samples - overlap_samples

    # Count total chunks required
    chunk_starts = list(range(0, total_samples, step_samples))
    total_chunks = max(1, len(chunk_starts))

    if total_samples <= chunk_samples:
        if telemetry_callback:
            telemetry_callback(1, 1, 0.0, "Diffusing single 5.12s window...")
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
        waveform = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
        waveform = np.squeeze(waveform).astype(np.float32)
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = (waveform / peak) * 0.95
        sf.write(output_wav, waveform, 48000)
        if telemetry_callback:
            telemetry_callback(1, 1, 100.0, "Super-resolution diffusion complete.")
        return output_wav

    out_sr = 48000
    out_chunk_samples = int(chunk_duration * out_sr)
    out_overlap_samples = int(overlap_duration * out_sr)
    out_step_samples = out_chunk_samples - out_overlap_samples

    expected_out_len = int(np.ceil((total_samples / in_sr) * out_sr)) + out_overlap_samples
    stitched_output = np.zeros(expected_out_len, dtype=np.float32)
    weight_envelope = np.zeros(expected_out_len, dtype=np.float32)

    fade_in = np.linspace(0.0, 1.0, out_overlap_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, out_overlap_samples, dtype=np.float32)

    chunk_window = np.ones(out_chunk_samples, dtype=np.float32)
    chunk_window[:out_overlap_samples] = fade_in
    chunk_window[-out_overlap_samples:] = fade_out

    temp_chunk_path = Path(input_wav).parent / "temp_audiosr_chunk.wav"
    cur_out_idx = 0

    for chunk_idx, cur_in_idx in enumerate(chunk_starts):
        in_end = min(cur_in_idx + chunk_samples, total_samples)
        segment = audio_data[cur_in_idx:in_end]
        actual_len = len(segment)

        if actual_len < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - actual_len))

        sf.write(str(temp_chunk_path), segment, in_sr)

        pct = round((chunk_idx / total_chunks) * 100, 1)
        if telemetry_callback:
            telemetry_callback(
                chunk_idx + 1,
                total_chunks,
                pct,
                f"Diffusing chunk {chunk_idx + 1}/{total_chunks} ({pct}% complete)..."
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            chunk_wave = super_resolution(
                model,
                str(temp_chunk_path),
                seed=int(seed),
                guidance_scale=float(guidance_scale),
                ddim_steps=int(ddim_steps),
            )

        chunk_wave = chunk_wave.detach().cpu().numpy() if isinstance(chunk_wave, torch.Tensor) else chunk_wave
        chunk_wave = np.squeeze(chunk_wave).astype(np.float32)

        actual_out_len = min(len(chunk_wave), int(round((actual_len / in_sr) * out_sr)))

        active_window = chunk_window[:actual_out_len].copy()
        if cur_in_idx == 0:
            active_window[:out_overlap_samples] = 1.0
        if in_end >= total_samples:
            active_window[-out_overlap_samples:] = 1.0

        end_out_idx = cur_out_idx + actual_out_len
        if end_out_idx > len(stitched_output):
            stitched_output = np.pad(stitched_output, (0, end_out_idx - len(stitched_output)))
            weight_envelope = np.pad(weight_envelope, (0, end_out_idx - len(weight_envelope)))

        stitched_output[cur_out_idx:end_out_idx] += chunk_wave[:actual_out_len] * active_window
        weight_envelope[cur_out_idx:end_out_idx] += active_window

        cur_out_idx += out_step_samples

    if temp_chunk_path.exists():
        temp_chunk_path.unlink()

    non_zero = weight_envelope > 1e-4
    stitched_output[non_zero] /= weight_envelope[non_zero]

    target_final_len = int(round((total_samples / in_sr) * out_sr))
    stitched_output = stitched_output[:target_final_len]

    peak = np.max(np.abs(stitched_output))
    if peak > 0:
        stitched_output = (stitched_output / peak) * 0.95

    sf.write(output_wav, stitched_output, out_sr)

    if telemetry_callback:
        telemetry_callback(total_chunks, total_chunks, 100.0, "Super-resolution 48kHz synthesis complete.")

    return output_wav


def synthesize_sentence_blocks(
    sentence_blocks: list,
    ref_audio_path: str,
    run_dir: Path,
    temperature: float = DEFAULT_TEMPERATURE,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    seed: int = 42,
    cue_cache_dir: Optional[Path] = None,
    progress_callback: Optional[Callable] = None,
) -> Tuple[str, List[Dict]]:
    """
    Renders speech blocks with per-cue checkpointing.
    Skips individual lines that have already been synthesized in cue_cache_dir.
    """
    if cue_cache_dir is None:
        cue_cache_dir = run_dir / "cues"
    cue_cache_dir.mkdir(parents=True, exist_ok=True)

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

        next_start_sec = (
            sentence_blocks[idx + 1]["start"]
            if idx + 1 < len(sentence_blocks)
            else s_end + 1.0
        )
        available_window = max(0.5, next_start_sec - s_start - 0.08)
        max_samples = int(available_window * SAMPLE_RATE)
        start_sample = max(int(s_start * SAMPLE_RATE), last_end_sample)
        timecode_str = f"{s_start:05.2f}s - {s_end:05.2f}s"

        cached_cue_wav = cue_cache_dir / f"cue_{cue_num:03d}.wav"

        if is_valid_file(cached_cue_wav):
            clip, _ = sf.read(str(cached_cue_wav), dtype="float32")
            stretch = 1.0
            status_tag = "Cached (Skipped)"
            raw_duration = len(clip) / SAMPLE_RATE
            peak_val = calculate_metrics(clip)
        else:
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
            sf.write(str(cached_cue_wav), clip, SAMPLE_RATE)

        sf.write(temp_preview_path, clip, SAMPLE_RATE)
        
        clip = apply_micro_fades(clip, SAMPLE_RATE, fade_ms=6.0)

        sf.write(temp_preview_path, clip, SAMPLE_RATE)

        end_sample = start_sample + len(clip)
        if end_sample > len(master_timeline):
            master_timeline = np.pad(master_timeline, (0, end_sample - len(master_timeline)))

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
    """Slices into master_timeline, re-synthesizes ONLY the chosen cue, and updates the cache."""
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

    clip = apply_micro_fades(clip, SAMPLE_RATE, fade_ms=6.0)
    
    peak_val = calculate_metrics(clip)

    # Invalidate and overwrite the cue cache
    cue_cache_dir = run_dir / "cues"
    cue_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_cue_wav = cue_cache_dir / f"cue_{cue_index:03d}.wav"
    sf.write(str(cached_cue_wav), clip, SAMPLE_RATE)

    # Splice into existing timeline
    raw_wav_path = str(run_dir / "timeline_raw.wav")
    master_timeline, sr = sf.read(raw_wav_path, dtype="float32")

    end_sample = start_sample + len(clip)
    if end_sample > len(master_timeline):
        master_timeline = np.pad(master_timeline, (0, end_sample - len(master_timeline)))

    clear_window_end = min(len(master_timeline), int(next_start_sec * SAMPLE_RATE))
    master_timeline[start_sample:clear_window_end] = 0.0
    master_timeline[start_sample:end_sample] = clip

    peak = np.max(np.abs(master_timeline))
    if peak > 0:
        master_timeline = (master_timeline / peak) * 0.92

    sf.write(raw_wav_path, master_timeline, SAMPLE_RATE)

    final_audio = raw_wav_path
    if apply_eq:
        mastered_wav = str(run_dir / "timeline_mastered.wav")
        apply_broadcast_mastering(raw_wav_path, mastered_wav)
        final_audio = mastered_wav

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

def apply_micro_fades(clip: np.ndarray, sr: int = SAMPLE_RATE, fade_ms: float = 25.0) -> np.ndarray:
    """Removes DC bias and applies a 25ms Hann envelope to eliminate vocoder onset pops."""
    if len(clip) == 0:
        return clip

    # 1. Strip DC offset
    clip = clip - np.mean(clip)

    # 2. 25ms fade (600 samples at 24kHz) covers full vocoder onset/offset transients
    fade_len = int((fade_ms / 1000.0) * sr)
    fade_len = min(fade_len, len(clip) // 4)

    if fade_len > 0:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))

        clip[:fade_len] *= fade_in
        clip[-fade_len:] *= fade_out

    return clip.astype(np.float32)