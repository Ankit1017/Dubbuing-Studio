"""Main application entry point for the DAW-Grade Neural Dubbing Workstation."""

import concurrent.futures
import datetime
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Silence Windows Proactor Socket Drop Errors
if sys.platform == "win32":
    from asyncio.proactor_events import _ProactorBasePipeTransport
    orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    def silenced_call_connection_lost(self, exc):
        try:
            orig_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass
        except OSError as e:
            if getattr(e, "winerror", None) == 10054:
                pass
            else:
                raise

    _ProactorBasePipeTransport._call_connection_lost = silenced_call_connection_lost

from config import (
    DEFAULT_CFG_WEIGHT,
    DEFAULT_EXAGGERATION,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    OUTPUTS_DIR,
    VOICES_DIR,
)
from text_processing import build_sentence_blocks
from media_utils import (
    apply_broadcast_mastering,
    extract_audio_from_video,
    mix_voice_with_background,
    mux_video_universal,
    probe_video_metadata,
    separate_audio_stems,
)
from audio_engine import (
    get_vram_usage,
    is_valid_file,
    resynthesize_single_cue,
    super_resolve_timeline,
    synthesize_sentence_blocks,
    transcribe_video_audio,
)
from ui_components import (
    STUDIO_CSS,
    render_header_html,
    render_hud_html,
    render_wavesurfer_component,
)


def scan_voice_presets():
    valid_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    if not VOICES_DIR.exists():
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
    voices = [f.name for f in VOICES_DIR.iterdir() if f.suffix.lower() in valid_exts]
    return sorted(voices)


def scan_saved_runs():
    if not OUTPUTS_DIR.exists():
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [str(p) for p in sorted(OUTPUTS_DIR.glob("run_*"), reverse=True) if p.is_dir()]
    return runs


def load_selected_voice_preview(selected_voice):
    if not selected_voice:
        return None
    voice_path = VOICES_DIR / selected_voice
    return str(voice_path) if voice_path.exists() else None


def run_pipeline(
    run_mode,
    resume_target_dir,
    video_file,
    voice_preset_name,
    custom_voice_file,
    temp,
    exag,
    cfg_w,
    seed_val,
    apply_eq,
    enable_audiosr,
):
    start_time = time.time()

    # 1. Resolve Workspace Directory
    if run_mode == "Resume Existing Run" and resume_target_dir and Path(resume_target_dir).is_dir():
        run_dir = Path(resume_target_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUTS_DIR / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    cue_cache_dir = run_dir / "cues"
    cue_cache_dir.mkdir(parents=True, exist_ok=True)

    # 2. Resolve Inputs & Meta
    session_meta_path = run_dir / "session_meta.json"
    session_meta = {}
    if is_valid_file(session_meta_path):
        try:
            with open(session_meta_path, "r", encoding="utf-8") as f:
                session_meta = json.load(f)
        except Exception:
            session_meta = {}

    if video_file is not None:
        video_path = video_file.name if hasattr(video_file, "name") else str(video_file)
        session_meta["video_path"] = video_path
    elif "video_path" in session_meta and is_valid_file(session_meta["video_path"]):
        video_path = session_meta["video_path"]
    else:
        raise gr.Error("Missing input video. Provide an MP4, MOV, or MKV file.")

    selected_voice_path = None
    if custom_voice_file is not None:
        selected_voice_path = str(custom_voice_file)
    elif voice_preset_name:
        preset_file = VOICES_DIR / voice_preset_name
        if preset_file.exists():
            selected_voice_path = str(preset_file)
    elif "voice_path" in session_meta and is_valid_file(session_meta["voice_path"]):
        selected_voice_path = session_meta["voice_path"]

    if not selected_voice_path:
        raise gr.Error("Missing voice reference. Select a preset or upload a WAV sample.")

    session_meta["voice_path"] = selected_voice_path
    session_meta["apply_eq"] = apply_eq
    session_meta["enable_audiosr"] = enable_audiosr

    with open(session_meta_path, "w", encoding="utf-8") as f:
        json.dump(session_meta, f, indent=2)

    logs = []
    def log(msg):
        t = f"{time.time() - start_time:05.1f}s"
        logs.append(f"[{t}] {msg}")
        return "\n".join(logs[-12:])

    # -------------------------------------------------------------
    # STAGE 1: DEMUX & DEMUCS STEM ISOLATION
    # -------------------------------------------------------------
    _, _, aspect_label = probe_video_metadata(video_path)
    extracted_audio = str(run_dir / "extracted_audio.wav")
    vocals_stem = str(run_dir / "vocals.wav")
    bg_stem = str(run_dir / "no_vocals.wav")

    yield (
        "STAGE 1/4: AUDIO DEMUX & STEMS",
        "Inspecting cached media stems or demuxing...",
        render_hud_html("0/0", "0.0s", get_vram_usage(), aspect_label),
        "Awaiting transcription stream...",
        pd.DataFrame(columns=["Cue", "Timestamp", "Script Segment", "Window", "Duration", "WSOLA", "Peak", "Status"]),
        None, None, None,
        render_wavesurfer_component(""),
        gr.update(choices=[]),
        {},
        log(f"Workspace: {run_dir.name}\nAspect detected: {aspect_label}")
    )

    if not is_valid_file(extracted_audio):
        extract_audio_from_video(video_path, extracted_audio)
        log("[✓] Audio stream extracted from video container.")
    else:
        log("[Skipped] Using existing extracted_audio.wav.")

    if not (is_valid_file(vocals_stem) and is_valid_file(bg_stem)):
        log("Engaging Demucs neural stem separator...")
        vocals_stem, bg_stem = separate_audio_stems(extracted_audio, run_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("[✓] Demucs stems generated.")
    else:
        log("[Skipped] Using existing isolated Demucs stems.")

    # -------------------------------------------------------------
    # STAGE 2: TRANSCRIPTION & ALIGNMENT
    # -------------------------------------------------------------
    yield (
        "STAGE 2/4: SPEECH ALIGNMENT",
        "Checking existing speech boundaries or running Whisper...",
        render_hud_html("0/0", f"{time.time() - start_time:.1f}s", get_vram_usage(), aspect_label),
        "Synchronizing phonetic cues...",
        pd.DataFrame(columns=["Cue", "Timestamp", "Script Segment", "Window", "Duration", "WSOLA", "Peak", "Status"]),
        None, None, None,
        render_wavesurfer_component(""),
        gr.update(choices=[]),
        {},
        log("Resolving subtitle alignment boundaries...")
    )

    srt_path = str(run_dir / "aligned_subtitles.srt")
    cues_json_path = run_dir / "sentence_blocks.json"

    if is_valid_file(cues_json_path) and is_valid_file(srt_path):
        with open(cues_json_path, "r", encoding="utf-8") as f:
            sentence_blocks = json.load(f)
        log(f"[Skipped] Loaded {len(sentence_blocks)} aligned sentence blocks from disk.")
    else:
        _, raw_cues = transcribe_video_audio(vocals_stem, srt_path)
        if not raw_cues:
            raise gr.Error("No clear dialogue detected in the source video.")
        sentence_blocks = build_sentence_blocks(srt_path)
        with open(cues_json_path, "w", encoding="utf-8") as f:
            json.dump(sentence_blocks, f, indent=2)
        log(f"[✓] Transcribed and aggregated {len(sentence_blocks)} speech cues.")

    total_sentences = len(sentence_blocks)
    choices = [f"Cue {i+1}: {b['text'][:25]}..." for i, b in enumerate(sentence_blocks)]

    # -------------------------------------------------------------
    # STAGE 3: CHATTERBOX TTS (Per-Cue Checkpointing)
    # -------------------------------------------------------------
    yield (
        "STAGE 3/4: ACOUSTIC SYNTHESIS",
        f"Checking checkpointed cues for {total_sentences} sentences...",
        render_hud_html(f"0/{total_sentences}", f"{time.time() - start_time:.1f}s", get_vram_usage(), aspect_label),
        "Conditioning voice profile and synthesizing uncached cues...",
        pd.DataFrame(columns=["Cue", "Timestamp", "Script Segment", "Window", "Duration", "WSOLA", "Peak", "Status"]),
        None, None, None,
        render_wavesurfer_component(""),
        gr.update(choices=choices),
        {},
        log("Synthesizing missing dialogue cues in parallel buffer...")
    )

    active_state = {"records": []}

    def on_sentence_synthesized(cue_num, timecode_str, raw_text, records, preview_path, peak_val, stretch):
        active_state["records"] = records
        active_state["latest_yield"] = (
            "STAGE 3/4: ACOUSTIC SYNTHESIS",
            f"Synthesizing [{cue_num}/{total_sentences}]",
            render_hud_html(f"{cue_num}/{total_sentences}", f"{time.time() - start_time:.1f}s", get_vram_usage(), aspect_label),
            f"[{timecode_str}]\n\"{raw_text}\"",
            pd.DataFrame(records),
            preview_path, None, None,
            render_wavesurfer_component(""),
            gr.update(choices=choices),
            {},
            log(f"Sentence #{cue_num} ({peak_val} | WSOLA {stretch:.2f}x)")
        )

    def run_synthesis():
        return synthesize_sentence_blocks(
            sentence_blocks=sentence_blocks,
            ref_audio_path=selected_voice_path,
            run_dir=run_dir,
            temperature=temp,
            exaggeration=exag,
            cfg_weight=cfg_w,
            seed=seed_val,
            cue_cache_dir=cue_cache_dir,
            progress_callback=on_sentence_synthesized
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_synthesis)
        while not future.done():
            if "latest_yield" in active_state:
                yield active_state["latest_yield"]
            time.sleep(0.1)
        raw_wav_path, final_records = future.result()

    records_json_path = run_dir / "records.json"
    with open(records_json_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2)

    # -------------------------------------------------------------
    # STAGE 3.5: AUDIOSR SUPER-RESOLUTION (Optional)
    # -------------------------------------------------------------
    timeline_48k = str(run_dir / "timeline_48k.wav")
    final_audio = raw_wav_path

    if not enable_audiosr:
        log("[Skipped] AudioSR disabled by user. Retaining 24kHz raw speech.")
        yield (
            "STAGE 3.5: SUPER-RESOLUTION (BYPASSED)",
            "AudioSR neural diffusion bypassed by configuration.",
            render_hud_html(f"{total_sentences}/{total_sentences}", f"{time.time() - start_time:.1f}s", get_vram_usage(), aspect_label),
            "Passing 24kHz speech directly to mastering chain...",
            pd.DataFrame(final_records),
            None, None, None,
            render_wavesurfer_component(""),
            gr.update(choices=choices),
            {},
            log("AudioSR bypassed. Proceeding directly to mastering and muxing.")
        )
    elif is_valid_file(timeline_48k):
        final_audio = timeline_48k
        log("[Skipped] Using existing 48kHz AudioSR master timeline.")
        yield (
            "STAGE 3.5: NEURAL SUPER-RESOLUTION",
            "Loaded pre-existing 48,000 Hz timeline from disk.",
            render_hud_html(f"{total_sentences}/{total_sentences}", f"{time.time() - start_time:.1f}s", get_vram_usage(), aspect_label),
            "High-frequency 48kHz audio validated.",
            pd.DataFrame(final_records),
            None, None, None,
            render_wavesurfer_component(""),
            gr.update(choices=choices),
            {},
            log("[Skipped] 48kHz AudioSR master previously completed.")
        )
    else:
        audiosr_state = {}

        def on_audiosr_chunk(cur_c, total_c, pct, status_str):
            audiosr_state["latest_yield"] = (
                "STAGE 3.5: NEURAL SUPER-RESOLUTION",
                f"Diffusing AudioSR Chunk [{cur_c}/{total_c}] ({pct:.1f}%)",
                render_hud_html(f"{total_sentences}/{total_sentences}", f"{time.time() - start_time:.1f}s", get_vram_usage(), aspect_label),
                status_str,
                pd.DataFrame(final_records),
                None, None, None,
                render_wavesurfer_component(""),
                gr.update(choices=choices),
                {},
                log(f"[AudioSR] Chunk {cur_c}/{total_c} ({pct:.1f}%) -> {status_str}")
            )

        def run_audiosr_task():
            return super_resolve_timeline(
                input_wav=raw_wav_path,
                output_wav=timeline_48k,
                seed=seed_val,
                ddim_steps=20,
                telemetry_callback=on_audiosr_chunk
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                sr_future = executor.submit(run_audiosr_task)
                while not sr_future.done():
                    if "latest_yield" in audiosr_state:
                        yield audiosr_state["latest_yield"]
                    time.sleep(0.15)
                final_audio = sr_future.result()
            log("[✓] AudioSR super-resolution complete (48,000 Hz master produced).")
        except Exception as e:
            log(f"[!] AudioSR bypassed due to error ({e}). Using 24kHz raw speech.")
            final_audio = raw_wav_path

    # -------------------------------------------------------------
    # STAGE 4: MASTERING & MUXING
    # -------------------------------------------------------------
    yield (
        "STAGE 4/4: MASTERING & MUXING",
        "Applying de-essing, dynamic compression, and -14 LUFS loudness...",
        render_hud_html(f"{total_sentences}/{total_sentences}", f"{time.time() - start_time:.1f}s", get_vram_usage(), aspect_label),
        "Rendering web-compatible MP4 container...",
        pd.DataFrame(final_records),
        None, None, None,
        render_wavesurfer_component(""),
        gr.update(choices=choices),
        {},
        log("Mastering audio buffers and muxing final video container...")
    )

    if apply_eq:
        mastered_wav = str(run_dir / "timeline_mastered.wav")
        if not is_valid_file(mastered_wav):
            apply_broadcast_mastering(final_audio, mastered_wav)
            log("[✓] Broadcast mastering applied (-14 LUFS).")
        else:
            log("[Skipped] Mastered timeline audio already exists.")
        final_audio = mastered_wav

    if is_valid_file(bg_stem):
        bg_data, _ = sf.read(bg_stem, dtype="float32")
        bg_rms = np.sqrt(np.mean(bg_data ** 2))
        
        # If RMS is above -45 dBFS, genuine music or ambiance is present
        if bg_rms > 0.005:
            mixed_master = str(run_dir / "master_mixed_timeline.wav")
            if not is_valid_file(mixed_master):
                final_audio = mix_voice_with_background(final_audio, bg_stem, mixed_master)
                log("[✓] Mixed voice with original background music stem.")
            else:
                final_audio = mixed_master
        else:
            log("[Skipped] Video has no background music; skipping Demucs stem mix.")

    final_video = str(run_dir / "final_mastered.mp4")
    if not is_valid_file(final_video):
        mux_video_universal(video_path, final_audio, final_video)
        log(f"[✓] Final video muxed to {final_video}")
    else:
        log("[Skipped] Master video container already built.")

    total_elapsed = f"{time.time() - start_time:.1f}s"
    session_data = {
        "run_dir": str(run_dir),
        "sentence_blocks": sentence_blocks,
        "records": final_records,
        "ref_audio": selected_voice_path,
        "video_path": video_path,
        "apply_eq": apply_eq,
    }

    # -------------------------------------------------------------
    # SYSTEM IDLE: COMPLETE (12 Components Active)
    # -------------------------------------------------------------
    yield (
        "SYSTEM IDLE: COMPLETE",
        f"Master presentation rendered in {total_elapsed}",
        render_hud_html(f"{total_sentences}/{total_sentences}", total_elapsed, get_vram_usage(), aspect_label),
        "Pipeline complete. Play master video or inspect individual cues below.",
        pd.DataFrame(final_records),
        final_audio, final_video, srt_path,
        render_wavesurfer_component(f"/file={os.path.abspath(final_audio)}"),
        gr.update(choices=choices, value=choices[0] if choices else None),
        session_data,
        log(f"Render complete. Workspace ready at {run_dir.resolve()}")
    )


# --- Gradio Interface ---
with gr.Blocks(title="Neural Studio DAW") as demo:
    gr.HTML(render_header_html())

    session_state = gr.State({})

    with gr.Row():
        # LEFT CONTROL RACK
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("##### ⚙️ WORKSPACE & CHECKPOINTING")
                run_mode = gr.Radio(
                    choices=["New Run", "Resume Existing Run"],
                    value="New Run",
                    label="Pipeline Execution Mode"
                )
                saved_runs = scan_saved_runs()
                run_selector = gr.Dropdown(
                    choices=saved_runs,
                    value=saved_runs[0] if saved_runs else None,
                    label="📂 Select Workspace to Resume",
                    interactive=True,
                    visible=False
                )

            with gr.Group():
                gr.Markdown("##### 🎛️ MEDIA INGEST")
                video_input = gr.File(
                    label="📁 Input Video File (.mp4, .mov, .mkv)",
                    file_types=[".mp4", ".mov", ".mkv", ".avi"],
                    type="filepath"
                )

                existing_voices = scan_voice_presets()
                default_voice = existing_voices[0] if existing_voices else None

                with gr.Row():
                    voice_dropdown = gr.Dropdown(
                        choices=existing_voices,
                        value=default_voice,
                        label="📚 Voice Preset (assets/voices/)",
                        scale=3
                    )
                    refresh_voices_btn = gr.Button("🔄", scale=1)

                voice_preview_player = gr.Audio(
                    value=str(VOICES_DIR / default_voice) if default_voice else None,
                    label="🎧 Voice Preset Preview",
                    type="filepath",
                    interactive=False
                )

                voice_upload_input = gr.Audio(
                    label="🎙️ Or Upload Custom Voice Profile (.wav)",
                    type="filepath"
                )

            with gr.Group():
                gr.Markdown("##### 🎚️ GLOBAL DSP CONTROLS")
                with gr.Row():
                    temp_slider = gr.Slider(0.1, 0.7, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature (Harmonics)")
                    exag_slider = gr.Slider(0.0, 0.5, value=DEFAULT_EXAGGERATION, step=0.05, label="Exaggeration (Cadence)")
                with gr.Row():
                    cfg_slider = gr.Slider(0.3, 1.5, value=DEFAULT_CFG_WEIGHT, step=0.05, label="CFG Weight (Voice Clamp)")
                    seed_ctrl = gr.Number(value=DEFAULT_SEED, precision=0, label="Anchor Seed")
                with gr.Row():
                    eq_toggle = gr.Checkbox(value=True, label="Broadcast Mastering (-14 LUFS)")
                    audiosr_toggle = gr.Checkbox(value=False, label="AudioSR 48kHz Super-Res (Slower)")

            run_btn = gr.Button("⚡ INITIATE DUBBING PIPELINE", variant="primary", size="lg", elem_id="primary-dub-btn")

        # RIGHT STUDIO MONITOR & DAW WORKSTATION
        with gr.Column(scale=6):
            with gr.Row():
                status_header = gr.Textbox(label="PIPELINE PHASE", value="SYSTEM IDLE: READY", elem_id="status-badge", scale=2)
                operation_sub = gr.Textbox(label="ACTIVE TASK", value="Standing by for video input", scale=3)

            hud_display = gr.HTML(render_hud_html())
            wavesurfer_box = gr.HTML(render_wavesurfer_component(""))

            teleprompter = gr.Textbox(
                label="🗣️ REAL-TIME SUBTITLE TELEPROMPTER & TELEMETRY",
                value="No subtitle cues currently active.",
                lines=2,
                elem_id="teleprompter-box"
            )

            # IN-PLACE CUE INSPECTOR
            with gr.Group(elem_classes="editor-panel"):
                gr.Markdown("##### 🎛️ IN-PLACE CUE INSPECTOR & RE-SYNTHESIZER")
                with gr.Row():
                    cue_selector = gr.Dropdown(label="Select Cue to Inspect / Re-record", choices=[])
                    cue_timecode_info = gr.Textbox(label="Timecode Window", interactive=False)

                cue_text_editor = gr.Textbox(
                    label="Edit Phonetics / Script for Selected Cue",
                    lines=2,
                    interactive=True
                )

                with gr.Row():
                    cue_temp_slider = gr.Slider(0.1, 0.7, value=DEFAULT_TEMPERATURE, step=0.05, label="Cue Temperature")
                    cue_cfg_slider = gr.Slider(0.3, 1.5, value=DEFAULT_CFG_WEIGHT, step=0.05, label="Cue CFG")
                    cue_seed_ctrl = gr.Number(value=DEFAULT_SEED, precision=0, label="Cue Seed")

                patch_cue_btn = gr.Button("⚡ RE-SYNTHESIZE & PATCH THIS CUE", variant="secondary")

            with gr.Row():
                cue_audio_monitor = gr.Audio(label="🎧 Cue DSP Audio Monitor", type="filepath", interactive=False)
                srt_file_download = gr.File(label="📄 Exported Timing Script (.srt)", interactive=False)

            live_cue_table = gr.Dataframe(
                headers=["Cue", "Timestamp", "Script Segment", "Window", "Duration", "WSOLA", "Peak", "Status"],
                label="AUDIO DSP FRAME REGISTRY",
                interactive=False,
                wrap=True
            )

            diag_terminal = gr.Textbox(
                label="DIAGNOSTIC RUNTIME TELEMETRY",
                lines=3,
                elem_id="terminal-log",
                interactive=False
            )

            output_video_player = gr.Video(label="🏆 MASTERED PRESENTATION", interactive=False, elem_classes="video-preview-wrapper")

    # --- Callbacks ---
    def on_run_mode_change(mode):
        dirs = scan_saved_runs()
        return gr.update(visible=(mode == "Resume Existing Run"), choices=dirs, value=dirs[0] if dirs else None)

    run_mode.change(
        fn=on_run_mode_change,
        inputs=[run_mode],
        outputs=[run_selector]
    )

    def refresh_dropdown():
        voices = scan_voice_presets()
        val = voices[0] if voices else None
        return gr.update(choices=voices, value=val), load_selected_voice_preview(val)

    refresh_voices_btn.click(
        fn=refresh_dropdown,
        outputs=[voice_dropdown, voice_preview_player]
    )

    voice_dropdown.change(
        fn=load_selected_voice_preview,
        inputs=voice_dropdown,
        outputs=voice_preview_player
    )

    def on_cue_selected(selected_choice, state):
        if not selected_choice or "sentence_blocks" not in state:
            return "", ""
        idx = int(selected_choice.split(":")[0].replace("Cue", "").strip()) - 1
        block = state["sentence_blocks"][idx]
        tc = f"{block['start']:.2f}s - {block['end']:.2f}s"
        return block["text"], tc

    cue_selector.change(
        fn=on_cue_selected,
        inputs=[cue_selector, session_state],
        outputs=[cue_text_editor, cue_timecode_info]
    )

    def on_patch_cue(cue_choice, new_text, c_temp, c_cfg, c_seed, state):
        if not cue_choice or "run_dir" not in state:
            raise gr.Error("No active run available to patch. Complete a generation run first.")

        cue_idx = int(cue_choice.split(":")[0].replace("Cue", "").strip())
        run_dir = Path(state["run_dir"])

        video_out, audio_out, preview_wav, updated_record = resynthesize_single_cue(
            cue_index=cue_idx,
            sentence_blocks=state["sentence_blocks"],
            new_text=new_text,
            ref_audio_path=state["ref_audio"],
            run_dir=run_dir,
            video_path=state["video_path"],
            temperature=c_temp,
            exaggeration=DEFAULT_EXAGGERATION,
            cfg_weight=c_cfg,
            seed=c_seed,
            apply_eq=state["apply_eq"]
        )

        records = state["records"]
        records[cue_idx - 1] = updated_record
        state["records"] = records

        new_choices = [f"Cue {i+1}: {b['text'][:25]}..." for i, b in enumerate(state["sentence_blocks"])]
        active_choice = new_choices[cue_idx - 1]

        ws_html = render_wavesurfer_component(f"/file={os.path.abspath(audio_out)}?t={int(time.time())}")

        return (
            video_out,
            preview_wav,
            pd.DataFrame(records),
            ws_html,
            gr.update(choices=new_choices, value=active_choice),
            state,
            f"[✓] Cue #{cue_idx} re-synthesized, spliced, and remuxed into master video."
        )

    patch_cue_btn.click(
        fn=on_patch_cue,
        inputs=[cue_selector, cue_text_editor, cue_temp_slider, cue_cfg_slider, cue_seed_ctrl, session_state],
        outputs=[
            output_video_player,
            cue_audio_monitor,
            live_cue_table,
            wavesurfer_box,
            cue_selector,
            session_state,
            diag_terminal
        ]
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            run_mode,
            run_selector,
            video_input,
            voice_dropdown,
            voice_upload_input,
            temp_slider,
            exag_slider,
            cfg_slider,
            seed_ctrl,
            eq_toggle,
            audiosr_toggle,
        ],
        outputs=[
            status_header,
            operation_sub,
            hud_display,
            teleprompter,
            live_cue_table,
            cue_audio_monitor,
            output_video_player,
            srt_file_download,
            wavesurfer_box,
            cue_selector,
            session_state,
            diag_terminal,
        ]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=STUDIO_CSS,
        theme=gr.themes.Base(),
        inbrowser=True
    )