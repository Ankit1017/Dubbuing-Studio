"""UI presentation components, WaveSurfer canvas, and DAW styling."""

STUDIO_CSS = """
:root {
    --bg-dark: #090c10;
    --card-bg: #0d1117;
    --card-border: #21262d;
    --accent-blue: #38bdf8;
    --accent-cyan: #00f2fe;
    --accent-green: #22c55e;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --text-primary: #e6edf3;
    --text-muted: #8b949e;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "JetBrains Mono", monospace !important;
}

.gr-box, .gr-panel, div[data-testid="block-container"], .block {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
}

input[type="file"], .file-preview, .upload-container, .drop-target, .center {
    background-color: #161b22 !important;
    border-color: #30363d !important;
    color: var(--text-primary) !important;
}

.video-preview-wrapper video {
    max-height: 480px !important;
    width: auto !important;
    margin: 0 auto !important;
    border-radius: 6px;
}

/* --- HUD Stat Cards --- */
.hud-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 4px 0 12px 0;
}
.hud-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.hud-label {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}
.hud-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--accent-cyan);
    margin-top: 2px;
}

/* --- WaveSurfer DAW Canvas Rack --- */
.wavesurfer-rack {
    background: #030712;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.wavesurfer-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #161b22;
}
.ws-btn {
    background: #1e293b;
    border: 1px solid #334155;
    color: #e2e8f0;
    border-radius: 4px;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    cursor: pointer;
    font-weight: bold;
}
.ws-btn:hover {
    background: #334155;
    color: #38bdf8;
}
.ws-timecode {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #38bdf8;
    font-weight: bold;
    min-width: 140px;
}

/* --- In-Place Cue Editor Panel --- */
.editor-panel {
    border: 1px solid #0284c7 !important;
    background: #071322 !important;
    border-radius: 8px !important;
    padding: 14px !important;
}

/* --- Status Badges --- */
#status-badge input {
    font-size: 0.85rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em !important;
    color: #4ade80 !important;
    background: #052e16 !important;
    border: 1px solid #14532d !important;
    text-align: center;
}

#teleprompter-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #e0f2fe !important;
    background: #040914 !important;
    border: 1px solid #0284c7 !important;
    box-shadow: inset 0 0 12px rgba(2, 132, 199, 0.25) !important;
}

.gr-dataframe th {
    background-color: #161b22 !important;
    color: var(--accent-blue) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
}
.gr-dataframe td {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-size: 0.82rem !important;
    border-bottom: 1px solid #21262d !important;
}

#terminal-log textarea {
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 0.78rem !important;
    color: #67e8f9 !important;
    background: #030712 !important;
    border: 1px solid #1f2937 !important;
}

#primary-dub-btn {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    border: 1px solid #38bdf8 !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
}
"""


def render_header_html() -> str:
    return """
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 18px; border-bottom: 1px solid #21262d; margin-bottom: 16px; background: #0d1117; border-radius: 8px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.3rem;">⚡</span>
            <span style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.02em; color: #f0f6fc;">NEURAL DUBBING WORKSTATION (PRO DAW)</span>
        </div>
        <div style="display: flex; gap: 8px;">
            <span style="font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; background: #0c2d48; color: #38bdf8; border: 1px solid #1e40af;">WAVESURFER SCRUBBER</span>
            <span style="font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; background: #3b0764; color: #d8b4fe; border: 1px solid #7e22ce;">IN-PLACE CUE PATCHER</span>
        </div>
    </div>
    """


def render_hud_html(cues="0/0", elapsed="0.0s", vram="0 MB", format_tag="Detecting..."):
    return f"""
    <div class="hud-grid">
        <div class="hud-card">
            <span class="hud-label">SENTENCE UNITS</span>
            <span class="hud-value">{cues}</span>
        </div>
        <div class="hud-card">
            <span class="hud-label">ELAPSED TIME</span>
            <span class="hud-value">{elapsed}</span>
        </div>
        <div class="hud-card">
            <span class="hud-label">CUDA VRAM</span>
            <span class="hud-value">{vram}</span>
        </div>
        <div class="hud-card">
            <span class="hud-label">DETECTED ASPECT</span>
            <span class="hud-value" style="font-size: 0.95rem;">{format_tag}</span>
        </div>
    </div>
    """


def render_wavesurfer_component(audio_url: str = "") -> str:
    """Renders the WaveSurfer.js canvas with interactive playhead, timecode, and zoom."""
    if not audio_url:
        return """
        <div class="wavesurfer-rack">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #64748b; text-align: center; padding: 24px;">
                🌊 Master timeline waveform inactive. Synthesize or load audio to engage visual scrubbing.
            </div>
        </div>
        """

    return f"""
    <div class="wavesurfer-rack">
        <div id="waveform" style="border-radius: 4px; overflow: hidden;"></div>
        <div class="wavesurfer-controls">
            <button class="ws-btn" onclick="window.wsPlayer.playPause()">▶ / ❚❚ PLAY/PAUSE</button>
            <button class="ws-btn" onclick="window.wsPlayer.stop()">⏹ STOP</button>
            <span class="ws-timecode" id="ws-time-display">00:00.00 / 00:00.00</span>
            <span style="font-size: 0.75rem; color: #64748b; margin-left: auto;">ZOOM:</span>
            <input type="range" min="10" max="150" value="30" oninput="window.wsPlayer.zoom(Number(this.value))" style="width: 100px;">
        </div>
        <script src="https://unpkg.com/wavesurfer.js@7"></script>
        <script>
            (function() {{
                if (window.wsPlayer) {{
                    window.wsPlayer.destroy();
                }}
                window.wsPlayer = WaveSurfer.create({{
                    container: '#waveform',
                    waveColor: '#1e3a8a',
                    progressColor: '#38bdf8',
                    cursorColor: '#00f2fe',
                    cursorWidth: 2,
                    height: 85,
                    barWidth: 2,
                    barGap: 2,
                    normalize: true,
                    url: '{audio_url}'
                }});

                const timeDisp = document.getElementById('ws-time-display');
                const fmt = (s) => {{
                    const m = Math.floor(s / 60);
                    const sec = Math.floor(s % 60);
                    const ms = Math.floor((s % 1) * 100);
                    return `${{String(m).padStart(2, '0')}}:${{String(sec).padStart(2, '0')}}.${{String(ms).padStart(2, '0')}}`;
                }};

                window.wsPlayer.on('audioprocess', (t) => {{
                    const dur = window.wsPlayer.getDuration() || 0;
                    timeDisp.innerText = `${{fmt(t)}} / ${{fmt(dur)}}`;
                }});
                window.wsPlayer.on('ready', () => {{
                    const dur = window.wsPlayer.getDuration() || 0;
                    timeDisp.innerText = `00:00.00 / ${{fmt(dur)}}`;
                }});
            }})();
        </script>
    </div>
    """