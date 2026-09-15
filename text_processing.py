"""Subtitle parsing, timestamp calculations, and linguistic pre-processing."""

import re
import pysrt
from config import TECH_LEXICON_RULES


def inject_prosody_punctuation(text: str) -> str:
    """
    Inserts micro-pauses and breath boundaries at transitional phrases
    and subordinating conjunctions to eliminate monotonic, breathless runs.
    """
    # 1. Transitional words that need a trailing pause when starting a thought
    leading_transitions = [
        "however", "therefore", "furthermore", "meanwhile", 
        "in addition", "as a result", "for example", "in fact"
    ]
    for word in leading_transitions:
        text = re.sub(rf"^(?i:\s*{word})(?!,)", f"{word.capitalize()},", text)

    # 2. Mid-sentence clause conjunctions that require a natural breath pause
    mid_conjunctions = [
        "however", "therefore", "meanwhile", "furthermore", 
        "which means", "because", "meaning that", "such as"
    ]
    for word in mid_conjunctions:
        # Match word if not already preceded by punctuation (. , ! ? -)
        pattern = rf"(?<![.,!?:;—])\s+\b({word})\b"
        text = re.sub(pattern, r", \1", text, flags=re.IGNORECASE)

    # 3. Clean up punctuation collisions and spacing
    text = re.sub(r"\s*,\s*,+", ", ", text)      # Collapse double commas
    text = re.sub(r"([.!?])\s*,", r"\1", text)   # Remove comma trailing a period/question mark
    text = re.sub(r"\s{2,}", " ", text)          # Collapse multi-spaces
    return text.strip()


def normalize_tech_script(text: str) -> str:
    """
    Normalizes technical terms, acronyms, and injects prosody punctuation 
    for natural speech cadence.
    """
    # Expand technical lexicon
    for pattern, replacement in TECH_LEXICON_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Inject breathing punctuation
    text = inject_prosody_punctuation(text)
    return text


def time_to_seconds(sub_time) -> float:
    """Converts a pysrt SubRipTime object to fractional seconds."""
    return (
        sub_time.hours * 3600
        + sub_time.minutes * 60
        + sub_time.seconds
        + sub_time.milliseconds / 1000.0
    )


def format_timestamp(seconds: float) -> str:
    """Converts seconds into standard SRT timestamp format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"


def build_sentence_blocks(srt_path: str):
    """
    Groups isolated subtitle cards into full grammatical sentences.
    Preserves start and end boundary timecodes of the overall clause.
    """
    subs = pysrt.open(srt_path, encoding="utf-8")
    blocks = []

    current_text = []
    current_start = None
    current_end = None

    for sub in subs:
        clean = re.sub(r"<[^>]+>", "", sub.text).strip().replace("\n", " ")
        if not clean:
            continue

        if current_start is None:
            current_start = time_to_seconds(sub.start)

        current_text.append(clean)
        current_end = time_to_seconds(sub.end)

        combined = " ".join(current_text)
        # Group cards until terminal punctuation or an adequate thought length is reached
        if clean.endswith((".", "!", "?", ":")) or len(combined.split()) >= 12:
            blocks.append({
                "start": current_start,
                "end": current_end,
                "text": combined
            })
            current_text = []
            current_start = None
            current_end = None

    if current_text:
        blocks.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_text)
        })

    return blocks