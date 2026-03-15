import json
import os
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="TA-RWARE Playback",
    page_icon="🤖",
    layout="wide",
)

# Manager URL reachable from the *browser* — must be the public-facing host:port,
# not the internal Docker name.  Override with MANAGER_PUBLIC_URL if needed.
MANAGER_URL = os.environ.get("MANAGER_URL", "http://manager:8001")
MANAGER_PUBLIC_URL = os.environ.get("MANAGER_PUBLIC_URL", MANAGER_URL)

st.title("🤖 TA-RWARE Warehouse Simulation")
st.caption("Each user gets their own isolated simulation instance")

# ── Per-session state ──────────────────────────────────────────────────────────
for key, default in [
    ("session_id", None),
    ("sim_url", None),
    ("running", False),
    ("stats", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    num_episodes = st.number_input("Number of Episodes", min_value=1, max_value=10000, value=3, step=1)

    st.subheader("🏭 Environment")
    size = st.selectbox("Warehouse Size", ["tiny", "small", "medium", "large", "extralarge"])
    col1, col2 = st.columns(2)
    with col1:
        n_agvs = st.number_input("AGVs", min_value=1, max_value=19, value=3)
    with col2:
        n_pickers = st.number_input("Pickers", min_value=1, max_value=9, value=2)
    obs = st.selectbox("Observability", ["partialobs", "globalobs"])
    env_name = f"tarware-{size}-{n_agvs}agvs-{n_pickers}pickers-{obs}-v1"
    st.caption(f"`{env_name}`")

    st.divider()

    start_button = st.button(
        "▶ Run Simulation", type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
    )
    if st.session_state.running:
        stop_button = st.button("⏹ Stop & Release", use_container_width=True)
    else:
        stop_button = False

    st.divider()
    st.subheader("📊 Stats")
    stat_episode    = st.empty()
    stat_pickrate   = st.empty()
    stat_return     = st.empty()
    stat_deliveries = st.empty()
    stat_fps        = st.empty()
    st.divider()
    status_box = st.empty()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id}`")
    elif st.session_state.running:
        st.caption("Session starting...")


def update_stats(stats: dict):
    ep    = stats.get("episode", "—")
    total = stats.get("num_episodes", "—")
    stat_episode.metric("Episode", f"{ep} / {total}")
    stat_pickrate.metric("Pick Rate (orders/hr)", stats.get("pick_rate", "—"))
    stat_return.metric("Global Return", stats.get("global_return", "—"))
    stat_deliveries.metric("Deliveries", stats.get("deliveries", "—"))
    stat_fps.metric("Sim FPS", stats.get("fps", "—"))


def wait_for_manager(timeout=30):
    status_box.info("⏳ Connecting to manager...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{MANAGER_URL}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                status_box.success(f"✅ Manager ready — {data['active_sessions']} active session(s)")
                return True
        except Exception:
            pass
        time.sleep(1)
    status_box.error("❌ Cannot reach manager service")
    return False


def create_session(num_ep: int, env: str) -> dict | None:
    while True:
        try:
            r = requests.post(
                f"{MANAGER_URL}/session",
                json={"num_episodes": num_ep, "env_name": env},
                timeout=15,
            )
            if r.status_code == 503 and r.json().get("detail") == "building":
                status_box.info("🔨 Simulation image is being built, please wait...")
                time.sleep(5)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            status_box.error(f"Failed to create session: {r.text}")
            return None
        except Exception as e:
            status_box.error(f"Failed to create session: {e}")
            return None


def destroy_session(session_id: str):
    try:
        requests.delete(f"{MANAGER_URL}/session/{session_id}", timeout=5)
    except Exception:
        pass


def render_live_player(session_id: str, manager_public_url: str, num_ep: int):
    """
    Render a self-contained HTML/JS player that polls the manager proxy
    directly from the browser — no Streamlit reruns needed.
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; color: #fafafa; background: transparent; padding: 12px; }}

  #status-bar {{
    font-size: 13px; color: #aaa; margin-bottom: 6px; min-height: 20px;
  }}
  #status-bar.done {{ color: #4caf50; }}
  #status-bar.error {{ color: #f44336; }}

  #layout {{
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 16px;
  }}

  @media (max-width: 700px) {{
    #layout {{ flex-direction: column; }}
    #controls-panel {{ width: 100% !important; }}
  }}

  #viewer {{
    flex: 0 0 auto;
  }}

  #viewer img {{
    max-width: 100%;
    max-height: 75vh;
    border-radius: 6px;
    display: block;
    background: #1a1a2e;
  }}
    max-width: 100%;
    max-height: 75vh;
    border-radius: 6px;
    display: block;
    background: #1a1a2e;
  }}

  #controls-panel {{
    width: 220px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding-top: 4px;
  }}

  #info {{ font-size: 13px; color: #aaa; }}

  #slider-row {{ display: flex; align-items: center; gap: 6px; }}
  #frame-slider {{ flex: 1; accent-color: #ff4b4b; min-width: 0; }}

  #controls {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
  button {{
    background: #262730; color: #fafafa; border: 1px solid #444;
    border-radius: 6px; padding: 5px 12px; cursor: pointer; font-size: 15px;
  }}
  button:hover {{ background: #3a3b45; }}
  button.active {{ background: #ff4b4b; border-color: #ff4b4b; }}
  button:disabled {{ opacity: 0.4; cursor: default; }}

  #fps-row {{ display: flex; align-items: center; gap: 6px; }}
  #fps-label {{ font-size: 13px; color: #aaa; }}
  #fps-input {{
    width: 55px; background: #262730; color: #fafafa;
    border: 1px solid #444; border-radius: 6px; padding: 4px 6px;
    font-size: 13px; text-align: center;
  }}

  #ep-buttons {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  #ep-buttons button {{ padding: 3px 8px; font-size: 12px; }}
  #ep-label {{ font-size: 12px; color: #888; margin-bottom: 2px; }}

  #header {{
    display: inline-flex;
    flex-direction: column;
    margin-bottom: 4px;
    min-width: 200px;
  }}
    height: 4px; background: #333; border-radius: 2px; overflow: hidden;
    margin-bottom: 10px;
  }}
  #loading-bar {{
    height: 100%; width: 0%; background: #ff4b4b;
    transition: width 0.4s ease;
  }}
</style>
</head>
<body>

<div id="header">
<div id="status-bar">⏳ Waiting for simulation to start...</div>
<div id="loading-bar-wrap"><div id="loading-bar"></div></div>
</div>

<div id="layout">
  <div id="viewer"><img id="frame-img" src="" alt="simulation frame" /></div>

  <div id="controls-panel">
    <div id="info">Frame <span id="frame-info">— / —</span></div>

    <div id="slider-row">
      <input type="range" id="frame-slider" min="0" max="0" value="0" step="1" />
    </div>

    <div id="controls">
      <button id="btn-prev" title="Previous frame">⏮</button>
      <button id="btn-play" title="Play/Pause">▶</button>
      <button id="btn-next" title="Next frame">⏭</button>
    </div>

    <div id="fps-row">
      <span id="fps-label">FPS:</span>
      <input type="number" id="fps-input" min="1" max="60" value="10" />
    </div>

    <div>
      <div id="ep-label">Jump to episode:</div>
      <div id="ep-buttons"></div>
    </div>
  </div>
</div>

<script>
  const SESSION_ID = "{session_id}";
  const MANAGER_URL = "{manager_public_url}";
  const NUM_EPISODES = {num_ep};
  const POLL_INTERVAL_MS = 2000;

  let frames = [];          // all b64 jpeg strings
  let boundaries = [];      // frame index where each episode starts
  let fetchedEpisodes = 0;
  let simDone = false;

  let currentIdx = 0;
  let playing = false;
  let intervalId = null;

  const img       = document.getElementById('frame-img');
  const slider    = document.getElementById('frame-slider');
  const frameInfo = document.getElementById('frame-info');
  const btnPlay   = document.getElementById('btn-play');
  const fpsInput  = document.getElementById('fps-input');
  const statusBar = document.getElementById('status-bar');
  const loadingBar = document.getElementById('loading-bar');
  const epContainer = document.getElementById('ep-buttons');

  // ── Player controls ───────────────────────────────────────────────────────

  function showFrame(idx) {{
    if (frames.length === 0) return;
    currentIdx = Math.max(0, Math.min(frames.length - 1, idx));
    img.src = 'data:image/jpeg;base64,' + frames[currentIdx];
    slider.value = currentIdx;
    frameInfo.textContent = `${{currentIdx + 1}} / ${{frames.length}}`;
  }}

  function getFps() {{
    return Math.max(1, Math.min(60, parseInt(fpsInput.value) || 10));
  }}

  function startPlaying() {{
    if (intervalId) clearInterval(intervalId);
    intervalId = setInterval(() => {{
      if (currentIdx >= frames.length - 1) {{
        if (simDone) {{ stopPlaying(); return; }}
        // sim still running — wait for more frames
        return;
      }}
      showFrame(currentIdx + 1);
    }}, 1000 / getFps());
    playing = true;
    btnPlay.textContent = '⏸';
    btnPlay.classList.add('active');
  }}

  function stopPlaying() {{
    if (intervalId) clearInterval(intervalId);
    intervalId = null;
    playing = false;
    btnPlay.textContent = '▶';
    btnPlay.classList.remove('active');
  }}

  function goTo(idx) {{
    stopPlaying();
    showFrame(idx);
  }}

  btnPlay.onclick = () => playing ? stopPlaying() : startPlaying();
  document.getElementById('btn-prev').onclick = () => goTo(currentIdx - 1);
  document.getElementById('btn-next').onclick = () => goTo(currentIdx + 1);
  slider.oninput = () => goTo(parseInt(slider.value));
  fpsInput.onchange = () => {{ if (playing) startPlaying(); }};

  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight') goTo(currentIdx + 1);
    else if (e.key === 'ArrowLeft') goTo(currentIdx - 1);
    else if (e.key === ' ') {{ e.preventDefault(); playing ? stopPlaying() : startPlaying(); }}
  }});

  // ── Episode button helpers ────────────────────────────────────────────────

  function addEpisodeButton(epIdx, boundary) {{
    const btn = document.createElement('button');
    btn.textContent = `Ep ${{epIdx + 1}}`;
    btn.onclick = () => goTo(boundary);
    epContainer.appendChild(btn);
  }}

  function appendFrames(newFrames, epIdx, boundary) {{
    const wasAtEnd = (currentIdx >= frames.length - 1) && frames.length > 0;
    frames = frames.concat(newFrames);
    boundaries.push(boundary);
    slider.max = frames.length - 1;
    addEpisodeButton(epIdx, boundary);

    // Show first frame as soon as we have any
    if (frames.length === newFrames.length) {{
      // This is the very first batch — skip frame 0 (possibly clipped)
      showFrame(Math.min(1, frames.length - 1));
    }} else if (wasAtEnd) {{
      // Was paused at end — advance into new episode
      showFrame(currentIdx + 1);
    }}

    // Update iframe height
    sendHeight();
  }}

  // ── Polling ───────────────────────────────────────────────────────────────

  async function fetchEpisode(epIdx) {{
    try {{
      const r = await fetch(`${{MANAGER_URL}}/sim/${{SESSION_ID}}/frames/${{epIdx}}`);
      if (r.status === 404) return null;   // not ready yet
      if (!r.ok) return null;
      const data = await r.json();
      return data.frames || null;
    }} catch (e) {{
      return null;
    }}
  }}

  async function pollStatus() {{
    try {{
      const r = await fetch(`${{MANAGER_URL}}/sim/${{SESSION_ID}}/status`);
      if (!r.ok) return null;
      return await r.json();
    }} catch (e) {{
      return null;
    }}
  }}

  async function tick() {{
    const status = await pollStatus();
    if (!status) {{
      statusBar.textContent = '⏳ Waiting for simulation container to start...';
      statusBar.className = '';
      setTimeout(tick, POLL_INTERVAL_MS);
      return;
    }}

    const completed = status.completed_episodes || 0;
    simDone = status.done || false;

    // Update progress bar
    const progress = simDone ? 100 : Math.round((completed / NUM_EPISODES) * 100);
    loadingBar.style.width = progress + '%';

    // Fetch all newly completed episodes
    while (fetchedEpisodes < completed) {{
      const epFrames = await fetchEpisode(fetchedEpisodes);
      if (epFrames === null) break;
      const boundary = frames.length;
      appendFrames(epFrames, fetchedEpisodes, boundary);
      fetchedEpisodes++;
    }}

    // Update status bar
    if (simDone && fetchedEpisodes >= completed) {{
      statusBar.textContent = `✅ Done — ${{frames.length}} frames across ${{boundaries.length}} episode(s)`;
      statusBar.className = 'done';
      loadingBar.style.width = '100%';
      // Auto-cleanup: release the sim container now that all frames are fetched
      fetch(`${{MANAGER_URL}}/session/${{SESSION_ID}}`, {{ method: 'DELETE' }}).catch(() => {{}});
      return; // stop polling
    }} else {{
      const ep = (status.stats && status.stats.episode !== undefined) ? status.stats.episode : fetchedEpisodes;
      statusBar.textContent = `⏳ Episode ${{ep + 1}} / ${{NUM_EPISODES}} running — ${{frames.length}} frames loaded`;
      statusBar.className = '';
    }}

    setTimeout(tick, POLL_INTERVAL_MS);
  }}

  // ── iframe height ─────────────────────────────────────────────────────────

  function sendHeight() {{
    const h = document.body.scrollHeight + 8;
    window.parent.postMessage({{type: 'streamlit:setFrameHeight', height: h}}, '*');
  }}

  img.onload = sendHeight;
  sendHeight();

  // ── Start ─────────────────────────────────────────────────────────────────
  tick();
</script>
</body>
</html>
"""
    components.html(html, height=900, scrolling=False)


# ── Main ───────────────────────────────────────────────────────────────────────
if not wait_for_manager():
    st.stop()

if stop_button and st.session_state.session_id:
    destroy_session(st.session_state.session_id)
    st.session_state.session_id = None
    st.session_state.sim_url = None
    st.session_state.running = False
    st.session_state.sim_started = False
    st.session_state.stats = {}
    status_box.info("Session released.")
    st.rerun()

if start_button and not st.session_state.running:
    if st.session_state.session_id:
        destroy_session(st.session_state.session_id)
    st.session_state.running = True
    st.session_state.sim_started = False
    st.session_state.stats = {}
    st.rerun()

if st.session_state.running and not st.session_state.get("sim_started"):
    st.session_state.sim_started = True
    status_box.info("🚀 Requesting simulation instance...")
    session = create_session(num_episodes, env_name)
    if session:
        st.session_state.session_id = session["session_id"]
        st.session_state.sim_url = session["sim_url"]
        status_box.success(f"✅ Simulation started — session `{session['session_id']}`")
    else:
        st.session_state.running = False
        st.session_state.sim_started = False
        status_box.error("❌ Failed to start simulation. Check the manager logs for details.")

# ── Render player (once session exists) ───────────────────────────────────────
if st.session_state.running and st.session_state.session_id:
    render_live_player(
        st.session_state.session_id,
        MANAGER_PUBLIC_URL,
        num_episodes,
    )
elif not st.session_state.running:
    st.info("👈 Configure and press **Run Simulation** to begin.")
