"""Central configuration for filesystem paths, audio DSP, and TTS hyperparameters."""

import os
from pathlib import Path

# --- Filesystem Layout ---
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
VOICES_DIR = ASSETS_DIR / "voices"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Auto-initialize project directories
for directory in [ASSETS_DIR, VOICES_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Audio Sampling & DSP Constants ---
SAMPLE_RATE = 24000
WHISPER_SAMPLE_RATE = 16000

# --- Default Chatterbox Hyperparameters ---
DEFAULT_TEMPERATURE = 0.42
DEFAULT_EXAGGERATION = 0.22
DEFAULT_CFG_WEIGHT = 0.68
DEFAULT_SEED = 42

# --- Video Muxing Filters ---
SCALE_FILTER = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

# --- Broadcast Mastering Chain ---
# Highpass (80Hz) -> Boxiness cut (320Hz) -> Speech presence (4.5kHz) -> De-esser -> Compressor -> -14 LUFS
MASTERING_FILTER_CHAIN = (
    "highpass=f=80,"
    "equalizer=f=320:t=q:w=1.5:g=-2.5,"
    "equalizer=f=4500:t=q:w=1.2:g=2.5,"
    "deesser=i=0.5:m=0.5:f=0.5,"
    "acompressor=threshold=-18dB:ratio=3:attack=15:release=120,"
    "loudnorm=I=-14:LRA=6:TP=-1.5"
)

# --- Technical Acronym & Version Number Normalization ---
TECH_LEXICON_RULES = [
    (r"\bHTTP\s*1\.1\b", "HTTP one point one"),
    (r"\bHTTP\s*1\.0\b", "HTTP one point zero"),
    (r"\bSYN-ACK\b", "SYN ACK"),
    (r"\bSYN\b", "SYN"),
    (r"\bACK\b", "ACK"),
    (r"\bJob\s*ID\b", "job I D"),
    (r"\bID:(\d+)\b", r"I D \1"),
    (r"\b(\d+)%\b", r"\1 percent"),
    (r"\bK8s\b", "Kubernetes"),
    (r"\bgRPC\b", "G R P C"),
    (r"\bAPIs?\b", "A P I"),
    (r"\b(\d+)\s*ms\b", r"\1 milliseconds"),
]

# --- AudioSR Neural Super-Resolution ---
AUDIOSR_MODEL_NAME = "speech"       # "speech" is fine-tuned specifically for voice/dialogue
AUDIOSR_DDIM_STEPS = 20             # 20 steps is optimal for speed on RTX 5060; 30-40 for max detail
AUDIOSR_GUIDANCE_SCALE = 3.5        # Balances natural vocal harmonic extension
DEFAULT_ENABLE_AUDIOSR = True       # Master toggle


# --- Reference Audio Conditioning Cleaners ---
ENABLE_REFERENCE_SANITIZATION = True
NOISE_REDUCE_PROP = 0.75       # 0.75 removes room hiss and reverb tails without thinning vocal formants
REFERENCE_NORMALIZE_PEAK = 0.95