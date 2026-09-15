dubbing_studio/
│
├── assets/
│   └── voices/           # Drop your reference narrator .wav files here
├── outputs/              # Every generation creates a self-contained timestamped run folder
│   └── run_YYYYMMDD_HHMMSS/
│       ├── extracted_audio.wav
│       ├── aligned_subtitles.srt
│       ├── timeline_raw.wav
│       ├── timeline_mastered.wav
│       └── final_mastered.mp4
├── config.py             # Paths, presets, mastering filters, and regex rules
├── text_processing.py    # Subtitle aggregation, timestamp conversion, and tech lexicon
├── media_utils.py        # FFmpeg wrapping (probing, demux, WSOLA stretch, mastering, muxing)
├── audio_engine.py       # Whisper & Chatterbox inference, collision prevention, VRAM telemetry
├── ui_components.py      # DAW CSS theme, HUD cards, and header layout
└── app.py                # Main Gradio application connecting UI to the modular pipeline
