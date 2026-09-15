"""FFmpeg wrapper for probing, audio extraction, mastering, and universal video muxing."""

import json
import os
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from config import MASTERING_FILTER_CHAIN, SCALE_FILTER

from pathlib import Path

def separate_audio_stems(audio_path: str, run_dir: Path) -> tuple[str, str]:
    """
    Separates input audio into isolated dialogue ('vocals.wav') 
    and ambient music/SFX ('no_vocals.wav') using Demucs.
    """
    stem_dir = run_dir / "demucs_stems"
    stem_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "-o", str(stem_dir),
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    track_name = Path(audio_path).stem
    demucs_out_dir = stem_dir / "htdemucs" / track_name
    
    vocals_path = str(demucs_out_dir / "vocals.wav")
    no_vocals_path = str(demucs_out_dir / "no_vocals.wav")
    return vocals_path, no_vocals_path


def mix_voice_with_background(dub_voice_path: str, bg_stem_path: str, mixed_out_path: str) -> str:
    """
    Mixes the newly synthesized voiceover with the original background audio.
    Applies gentle background volume attenuation (attenuated to -6dB / 0.5x).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", dub_voice_path,
        "-i", bg_stem_path,
        "-filter_complex",
        "[1:a]volume=0.5[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        mixed_out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mixed_out_path


def probe_video_metadata(video_path: str):
    """Detects resolution, orientation, and format aspect ratio."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_tags=rotate",
        "-of", "json",
        video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)["streams"][0]
        w, h = data.get("width", 1920), data.get("height", 1080)
        rotate = data.get("tags", {}).get("rotate", "0")

        if rotate in ["90", "270"]:
            w, h = h, w

        ratio_str = "9:16 Vertical" if h > w else "16:9 Landscape" if w > h else "1:1 Square"
        return w, h, f"{w}x{h} ({ratio_str})"
    except Exception:
        return 1920, 1080, "Auto-Detected"


def extract_audio_from_video(video_path: str, audio_out: str) -> str:
    """Extracts 16kHz mono PCM WAV from input video."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_out
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")
    return audio_out


def time_stretch_clean(audio: np.ndarray, rate: float, temp_dir: Path, sr: int = 24000) -> np.ndarray:
    """
    Pitch-preserving time stretch using FFmpeg's time-domain atempo filter.
    Avoids phase vocoder robotic comb-filtering.
    """
    if abs(rate - 1.0) < 0.03 or len(audio) == 0:
        return audio
    rate = max(0.85, min(rate, 1.25))

    temp_in = str(temp_dir / f"temp_str_in_{os.getpid()}.wav")
    temp_out = str(temp_dir / f"temp_str_out_{os.getpid()}.wav")
    sf.write(temp_in, audio, sr)

    cmd = ["ffmpeg", "-y", "-i", temp_in, "-filter:a", f"atempo={rate:.3f}", temp_out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    stretched, _ = sf.read(temp_out, dtype="float32")
    for f in (temp_in, temp_out):
        if os.path.exists(f):
            os.remove(f)
    return stretched


def calculate_metrics(clip: np.ndarray) -> str:
    """Calculates peak volume in dBFS."""
    if len(clip) == 0:
        return "-inf dBFS"
    peak = np.max(np.abs(clip))
    return f"{20 * np.log10(peak + 1e-9):.1f} dBFS"


def apply_broadcast_mastering(raw_wav: str, mastered_wav: str) -> str:
    """Applies highpass, mud cut, speech presence, de-essing, compression, and -14 LUFS normalizer."""
    cmd = ["ffmpeg", "-y", "-i", raw_wav, "-af", MASTERING_FILTER_CHAIN, mastered_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mastered_wav


def mux_video_universal(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Muxes dubbed audio into video container with even dimension padding and +faststart flag.
    Tries GPU NVENC first; falls back cleanly to CPU libx264.
    """
    cmd_nvenc = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", SCALE_FILTER,
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-preset", "p4",
        "-c:a", "aac",
        "-b:a", "256k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-movflags", "+faststart",
        "-shortest",
        output_path
    ]

    cmd_cpu = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", SCALE_FILTER,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "256k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-movflags", "+faststart",
        "-shortest",
        output_path
    ]

    mux_res = subprocess.run(cmd_nvenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if mux_res.returncode != 0:
        cpu_res = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cpu_res.returncode != 0:
            raise RuntimeError(f"FFmpeg muxing failed on both NVENC and CPU:\n{cpu_res.stderr}")

    return output_path