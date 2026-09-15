import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import OUTPUTS_DIR, VOICES_DIR, DEFAULT_TEMPERATURE, DEFAULT_CFG_WEIGHT, DEFAULT_SEED, DEFAULT_EXAGGERATION
from text_processing import build_sentence_blocks
from media_utils import probe_video_metadata, extract_audio_from_video, mux_video_universal, apply_broadcast_mastering
from audio_engine import transcribe_video_audio, synthesize_sentence_blocks, resynthesize_single_cue, get_vram_usage

app = FastAPI(title="Neural Dubbing Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated assets
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/assets/voices", StaticFiles(directory=str(VOICES_DIR)), name="voices")

SESSION_CACHE = {}

@app.get("/api/presets")
def get_presets():
    valid = (".wav", ".mp3", ".flac")
    return [f.name for f in VOICES_DIR.iterdir() if f.suffix.lower() in valid]

@app.post("/api/run-pipeline")
async def start_pipeline(
    video: UploadFile = File(...),
    voice_preset: str = Form(""),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    cfg_weight: float = Form(DEFAULT_CFG_WEIGHT),
    seed: int = Form(DEFAULT_SEED)
):
    import datetime
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = run_dir / video.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    ref_audio = VOICES_DIR / voice_preset

    # 1. Demux
    extracted = str(run_dir / "extracted_audio.wav")
    extract_audio_from_video(str(video_path), extracted)

    # 2. Transcribe & Group
    srt_out = str(run_dir / "aligned.srt")
    _, raw_cues = transcribe_video_audio(extracted, srt_out)
    blocks = build_sentence_blocks(srt_out)

    # 3. Synthesize
    raw_wav, records = synthesize_sentence_blocks(
        sentence_blocks=blocks,
        ref_audio_path=str(ref_audio),
        run_dir=run_dir,
        temperature=temperature,
        cfg_weight=cfg_weight,
        seed=seed
    )

    # 4. Master & Mux
    mastered = str(run_dir / "mastered.wav")
    apply_broadcast_mastering(raw_wav, mastered)
    final_video = str(run_dir / "final_master.mp4")
    mux_video_universal(str(video_path), mastered, final_video)

    SESSION_CACHE["latest"] = {
        "run_dir": str(run_dir),
        "sentence_blocks": blocks,
        "ref_audio": str(ref_audio),
        "video_path": str(video_path),
        "records": records
    }

    return {
        "status": "complete",
        "run_id": run_id,
        "video_url": f"/outputs/{run_id}/final_master.mp4",
        "src_audio_url": f"/outputs/{run_id}/extracted_audio.wav",
        "dub_audio_url": f"/outputs/{run_id}/mastered.wav",
        "cues": blocks,
        "records": records
    }

@app.post("/api/patch-cue")
async def patch_cue(
    cue_idx: int = Form(...),
    text: str = Form(...),
    temperature: float = Form(0.42),
    cfg_weight: float = Form(0.68),
    seed: int = Form(42)
):
    session = SESSION_CACHE.get("latest")
    if not session:
        return {"error": "No active session."}

    final_video, final_audio, preview_wav, updated_record = resynthesize_single_cue(
        cue_index=cue_idx,
        sentence_blocks=session["sentence_blocks"],
        new_text=text,
        ref_audio_path=session["ref_audio"],
        run_dir=Path(session["run_dir"]),
        video_path=session["video_path"],
        temperature=temperature,
        exaggeration=DEFAULT_EXAGGERATION,
        cfg_weight=cfg_weight,
        seed=seed,
        apply_eq=True
    )
    return {"status": "patched", "record": updated_record}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)