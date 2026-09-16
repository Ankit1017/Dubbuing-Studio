# Neural Studio DAW: Local AI Video Dubbing & Audio Restoration

An end-to-end, fully local AI-powered video dubbing and voice cloning workstation built in 100% Python. Designed to run locally on consumer NVIDIA RTX hardware, this studio isolates background music/SFX, transcribes and aligns speech cues via VAD, synthesizes cloned vocals using Chatterbox-TTS, eliminates voice collisions via non-overlapping timeline buffers, and upscales synthesized audio to 48 kHz broadcast quality using AudioSR diffusion.

## Key Architecture & Features

* **Multi-Stage Audio Pipeline:**

  1. **Stem Isolation (Demucs):** Extracts clean dialogue from background music and environmental sound effects.
  2. **VAD Speech Alignment (Faster-Whisper):** Generates timestamped cues with millisecond accuracy using voice activity detection.
  3. **Voice Cloning & Synthesis (Chatterbox-TTS):** Zero-shot vocal cloning from reference samples with normalized technical lexicon handling.
  4. **Collision-Safe Timeline Buffering:** Non-overlapping timeline slotting with lookahead window clamping to eliminate double-talk bleed.
  5. **WSOLA Time-Stretching:** Dynamic cadence compression (up to 1.25x) to fit cloned speech precisely into original video duration windows without pitch shifts.
  6. **Neural Super-Resolution (AudioSR):** 20-step DDIM latent diffusion upscales 24 kHz speech to 48 kHz studio masters, restoring 12 kHz–20 kHz high-frequency air.
  7. **Broadcast Mastering & Muxing:** Dynamic de-essing, multi-band compression, `-14 LUFS` loudness matching, and universal H.264 MP4 export via FFmpeg.
* **Format Agnostic:** Native support for both 9:16 vertical shorts/reels and standard 16:9 widescreen landscape videos.
* **Single-Cue In-Place Patching:** Re-synthesize, re-time, or tweak specific sentence cues without re-rendering the full video timeline.
* **Hardware Tuned:** Optimized for NVIDIA RTX GPUs (CUDA 12.8 / PyTorch `cu128`), featuring CPU-offloaded transcription to eliminate VRAM allocator contention.

## Directory Structure

**Plaintext**

```
dubbing_studio/
├── app.py                  # Gradio 6.0 Studio interface and streaming pipeline
├── audio_engine.py         # Whisper, Chatterbox-TTS, AudioSR, and audio patches
├── config.py               # Audio constants, sample rates, mastering filter chains
├── media_utils.py          # FFmpeg muxing, Demucs separation, WSOLA time-stretching
├── text_processing.py      # SRT parsing, technical lexicon normalization, timestamps
├── ui_components.py        # Custom DAW HUD cards, Wavesurfer HTML visualizer, CSS
├── assets/
│   └── voices/             # Reference narrator audio samples (.wav)
├── outputs/                # Timestamped runs, audio stems, mastered MP4s
└── requirements.txt        # Pinned dependency manifest
```

## Prerequisites & System Requirements

* **OS:** Windows 10/11 or Linux
* **Python:** 3.10
* **GPU:** NVIDIA GPU with **\$\\ge\$** 8 GB VRAM (RTX 3060 / 4060 / 5060 or higher)
* **CUDA:** 12.8 (or 12.6)
* **FFmpeg:** Installed and added to system `PATH`

## Installation

### 1. Clone the Repository

**PowerShell**

```
git clone https://github.com/Ankit1017/Dubbuing-Studio.git
cd dubbing_studio
```

### 2. Set Up Virtual Environment

**PowerShell**

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA 12.8

Install the matching CUDA wheels directly from PyTorch:

**PowerShell**

```
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Dependencies

**PowerShell**

```
pip install -r requirements.txt
pip install "numpy==1.26.4" "transformers==5.2.0"
```

## Critical Windows & Runtime Patches

This codebase contains native runtime patches embedded inside `audio_engine.py`:

1. **Perth Watermarker Bypass:** Replaces `resemble-perth`'s defect-ridden implicit watermarker to prevent `NoneType` crashes and preserve frequency bandwidth above 16.8 kHz.
2. **TorchCodec Windows Bypass:** Intercepts `torchaudio.load()` and `torchaudio.save()` to route via `soundfile`, bypassing missing FFmpeg C++ DLL requirements on Windows.
3. **CTranslate2 VRAM Decoupling:** Runs `faster-whisper` on CPU via `int8` quantization to avoid CUDA allocator memory collisions with PyTorch's Demucs stem separation.

## Usage

Start the local web workstation:

**PowerShell**

```
python app.py
```

Open your browser at `[http://127.0.0.1:7860](http://127.0.0.1:7860)`.

### Running a Dubbing Session

1. Upload your source video (`.mp4`, `.mkv`, or `.mov`).
2. Select or upload a target narrator reference audio clip (`.wav`).
3. Set your inference parameters:

   * **Temperature:**`0.70` (default)
   * **Exaggeration:**`0.50`
   * **CFG Weight:**`0.50`
   * **Seed:**`42` (ensures consistent vocal timbre across all cues)
4. Click **Run Master Dubbing Pipeline**.
5. Once completed, review the generated audio waveform, preview the muxed master video, or select individual cues from the dropdown to re-record specific lines in place.

## License

MIT License. Designed for local, privacy-focused video localization and speech synthesis.
