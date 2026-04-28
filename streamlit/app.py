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

_SIMULATOR_EXCLUDED_MAPS = {"full_dhl"}  # too large for live-playback; tuner can use it


@st.cache_data(ttl=60)
def get_maps() -> list[str]:
    try:
        r = requests.get(f"{MANAGER_URL}/maps", timeout=5)
        maps = r.json().get("maps", [])
        maps = [m for m in maps if m not in _SIMULATOR_EXCLUDED_MAPS]
        return maps if maps else ["tiny", "small", "medium", "large", "extralarge"]
    except Exception:
        return ["tiny", "small", "medium", "large", "extralarge"]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    num_episodes = st.number_input("Number of Episodes", min_value=1, max_value=10000, value=3, step=1)

    st.subheader("🏭 Environment")
    available_maps = get_maps()
    size = st.selectbox("Warehouse Map", available_maps)
    n_agvs = st.number_input("AGVs", min_value=1, max_value=19, value=3)
    n_pickers = st.number_input("Pickers", min_value=0, max_value=19, value=0)

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
    status_box = st.empty()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id}`")
    elif st.session_state.running:
        st.caption("Session starting...")


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


def create_session(num_ep: int, map_name: str, num_agvs: int, num_pickers: int) -> dict | None:
    while True:
        try:
            r = requests.post(
                f"{MANAGER_URL}/session",
                json={
                    "num_episodes": num_ep,
                    "map_name": map_name,
                    "num_agvs": num_agvs,
                    "num_pickers": num_pickers,
                },
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
    """Render a self-contained HTML/JS player + stats charts. All polling done
    client-side via fetch() to the manager proxy — no Streamlit reruns needed."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; color: #fafafa; background: transparent; padding: 12px; }}

  #status-bar {{ font-size: 13px; color: #aaa; margin-bottom: 6px; min-height: 20px; display: inline-block; }}
  #status-bar.done {{ color: #4caf50; }}
  #status-bar.error {{ color: #f44336; }}

  #header {{ display: inline-flex; flex-direction: column; margin-bottom: 4px; min-width: 200px; }}
  #loading-bar-wrap {{ height: 4px; background: #333; border-radius: 2px; overflow: hidden; margin-bottom: 10px; }}
  #loading-bar {{ height: 100%; width: 0%; background: #ff4b4b; transition: width 0.4s ease; }}

  #layout {{ display: flex; flex-direction: row; align-items: flex-start; gap: 16px; }}
  @media (max-width: 700px) {{
    #layout {{ flex-direction: column; }}
    #controls-panel {{ width: 100% !important; }}
  }}

  #viewer {{ flex: 0 0 auto; }}
  #viewer img {{ max-width: 100%; max-height: 75vh; border-radius: 6px; display: block; background: #1a1a2e; }}

  #controls-panel {{
    width: 220px; flex-shrink: 0; display: flex; flex-direction: column;
    gap: 12px; padding-top: 4px;
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

  /* ── Stats section ── */
  #stats-section {{ margin-top: 28px; }}
  #stats-title {{ font-size: 14px; font-weight: 600; color: #ccc; margin-bottom: 12px; }}
  #charts-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }}
  .chart-wrap {{
    background: #1e1e2e;
    border-radius: 8px;
    padding: 12px;
  }}
  .chart-label {{
    font-size: 11px;
    color: #888;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .chart-wrap canvas {{ width: 100% !important; height: 160px !important; }}

  .chart-details {{
    margin-top: 6px;
    font-size: 11px;
    color: #777;
  }}
  .chart-details summary {{
    cursor: pointer;
    color: #666;
    user-select: none;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .chart-details summary::before {{
    content: '▸';
    font-size: 9px;
    transition: transform 0.15s;
    display: inline-block;
  }}
  .chart-details[open] summary::before {{ transform: rotate(90deg); }}
  .chart-details summary::-webkit-details-marker {{ display: none; }}
  .chart-details p {{
    margin-top: 5px;
    padding: 6px 8px;
    background: #1e1e1e;
    border-left: 2px solid #444;
    border-radius: 0 4px 4px 0;
    line-height: 1.5;
    color: #999;
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

<div id="stats-section">
  <div id="stats-title">📊 Episode Statistics</div>
  <div id="charts-grid">
    <div class="chart-wrap">
      <div class="chart-label">Bin throughput (bins/hr)</div>
      <canvas id="chart-bin-throughput"></canvas>
      <details class="chart-details">
        <summary>How is this calculated?</summary>
        <p>deliveries &times; 3600 &divide; (seconds_per_step &times; timesteps). Each timestep represents one simulated step (4 seconds when TARWARE_STEPS_PER_SIMULATED_SECOND=0.25), so this scales the fleet's shelf delivery count to an hourly rate. AGV-side metric &mdash; bins delivered to the pickerwall, not items picked.</p>
      </details>
    </div>
    <div class="chart-wrap">
      <div class="chart-label">Global Return</div>
      <canvas id="chart-return"></canvas>
      <details class="chart-details">
        <summary>How is this calculated?</summary>
        <p>Sum of all reward signals across all bots for the episode. Each delivery to the pickerwall earns +1.0, each shelf pick-up or rack return earns +0.1, and every timestep deducts &minus;0.001 per bot.</p>
      </details>
    </div>
    <div class="chart-wrap">
      <div class="chart-label">Deliveries</div>
      <canvas id="chart-deliveries"></canvas>
      <details class="chart-details">
        <summary>How is this calculated?</summary>
        <p>Raw count of shelves successfully deposited at a pickerwall slot during the episode. One delivery = one requested shelf reaching the wall.</p>
      </details>
    </div>
    <div class="chart-wrap">
      <div class="chart-label">Distance / Bot</div>
      <canvas id="chart-dist-bot"></canvas>
      <details class="chart-details">
        <summary>How is this calculated?</summary>
        <p>Total grid cells moved by all bots divided by the number of bots. A move is counted each timestep a bot's position changes. Higher values mean bots are travelling further per episode.</p>
      </details>
    </div>
    <div class="chart-wrap">
      <div class="chart-label">Bot Utilisation %</div>
      <canvas id="chart-agv-util"></canvas>
      <details class="chart-details">
        <summary>How is this calculated?</summary>
        <p>100 &times; (1 &minus; idle_steps &divide; (n_bots &times; timesteps)). A bot is counted as idle when the environment marks it inactive. 100% means every bot was busy every step.</p>
      </details>
    </div>
    <div class="chart-wrap">
      <div class="chart-label">Clash Rate (%/step)</div>
      <canvas id="chart-clashes"></canvas>
      <details class="chart-details">
        <summary>How is this calculated?</summary>
        <p>(collision_events &divide; timesteps) &times; 100. A clash is recorded when two bots attempt to occupy the same cell in the same step. The environment resolves conflicts automatically.</p>
      </details>
    </div>
  </div>
</div>

<script>
  const SESSION_ID = "{session_id}";
  const MANAGER_URL = "{manager_public_url}";
  const NUM_EPISODES = {num_ep};
  const POLL_INTERVAL_MS = 2000;

  let frames = [];
  let boundaries = [];
  let fetchedEpisodes = 0;       // count of episodes whose frames are fully fetched
  let epFrameOffsets = {{}};       // per-episode within-episode offset already fetched
  let simDone = false;
  let currentIdx = 0;
  let currentEpisode = 0;
  let playing = false;
  let intervalId = null;
  let episodeStats = [];
  let charts = {{}};

  const img       = document.getElementById('frame-img');
  const slider    = document.getElementById('frame-slider');
  const frameInfo = document.getElementById('frame-info');
  const btnPlay   = document.getElementById('btn-play');
  const fpsInput  = document.getElementById('fps-input');
  const statusBar = document.getElementById('status-bar');
  const loadingBar = document.getElementById('loading-bar');
  const epContainer = document.getElementById('ep-buttons');

  // ── Player ────────────────────────────────────────────────────────────────

  function getEpisodeForFrame(idx) {{
    let ep = 0;
    for (let i = 0; i < boundaries.length; i++) {{
      if (boundaries[i] <= idx) ep = i;
      else break;
    }}
    return ep;
  }}

  function showFrame(idx) {{
    if (frames.length === 0) return;
    currentIdx = Math.max(0, Math.min(frames.length - 1, idx));
    img.src = 'data:image/jpeg;base64,' + frames[currentIdx];
    slider.value = currentIdx;
    frameInfo.textContent = `${{currentIdx + 1}} / ${{frames.length}}`;
    const ep = getEpisodeForFrame(currentIdx);
    if (ep !== currentEpisode) {{
      currentEpisode = ep;
      updateChartHighlight(ep);
    }}
  }}

  function getFps() {{ return Math.max(1, Math.min(60, parseInt(fpsInput.value) || 10)); }}

  function startPlaying() {{
    if (intervalId) clearInterval(intervalId);
    intervalId = setInterval(() => {{
      if (currentIdx >= frames.length - 1) {{
        if (simDone) {{ stopPlaying(); return; }}
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

  function goTo(idx) {{ stopPlaying(); showFrame(idx); }}

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

  function addEpisodeButton(epIdx, boundary) {{
    const btn = document.createElement('button');
    btn.textContent = `Ep ${{epIdx + 1}}`;
    btn.onclick = () => goTo(boundary);
    epContainer.appendChild(btn);
  }}

  function appendFrames(newFrames, epIdx, boundary, isContinuation=false) {{
    const wasAtEnd = (currentIdx >= frames.length - 1) && frames.length > 0;
    frames = frames.concat(newFrames);
    if (!isContinuation) {{
      boundaries.push(boundary);
      addEpisodeButton(epIdx, boundary);
    }}
    slider.max = Math.max(0, frames.length - 1);
    if (frames.length === newFrames.length) {{
      showFrame(Math.min(1, frames.length - 1));
    }} else if (wasAtEnd) {{
      showFrame(currentIdx + 1);
    }}
  }}

  // ── Charts ────────────────────────────────────────────────────────────────

  const CHART_DEFS = [
    {{ id: 'chart-bin-throughput',  key: 'bin_throughput',     label: 'Bins/hr'           }},
    {{ id: 'chart-return',     key: 'global_return',       label: 'Global Return'     }},
    {{ id: 'chart-deliveries', key: 'deliveries',          label: 'Deliveries'        }},
    {{ id: 'chart-dist-bot',   key: 'distance_per_bot',    label: 'Distance / Bot'    }},
    {{ id: 'chart-agv-util',   key: 'agv_utilization',     label: 'Bot Utilisation %' }},
    {{ id: 'chart-clashes',    key: 'clash_rate',          label: 'Clash Rate'        }},
  ];

  const BASE_COLOR   = 'rgba(255, 75, 75, 0.85)';
  const LINE_COLOR   = 'rgba(255, 75, 75, 0.4)';
  const HL_COLOR     = '#ffffff';

  function makeChart(canvasId, label) {{
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {{
      type: 'scatter',
      data: {{
        datasets: [
          {{
            label,
            data: [],
            borderColor: LINE_COLOR,
            backgroundColor: BASE_COLOR,
            pointRadius: 4,
            pointHoverRadius: 6,
            showLine: true,
            tension: 0.3,
            borderWidth: 1.5,
          }},
          {{
            label: 'current',
            data: [],
            borderColor: HL_COLOR,
            backgroundColor: HL_COLOR,
            pointRadius: 7,
            pointHoverRadius: 8,
            showLine: false,
          }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{
            title: {{ display: true, text: 'Episode', color: '#666', font: {{ size: 10 }} }},
            ticks: {{ color: '#666', font: {{ size: 10 }}, stepSize: 1 }},
            grid: {{ color: '#2a2a3a' }},
          }},
          y: {{
            ticks: {{ color: '#666', font: {{ size: 10 }} }},
            grid: {{ color: '#2a2a3a' }},
          }},
        }},
      }},
    }});
  }}

  function initCharts() {{
    CHART_DEFS.forEach(def => {{
      charts[def.key] = makeChart(def.id, def.label);
    }});
  }}

  function updateCharts(stats) {{
    CHART_DEFS.forEach(def => {{
      const chart = charts[def.key];
      if (!chart) return;
      chart.data.datasets[0].data = stats.map((s, i) => ({{ x: i + 1, y: s[def.key] }}));
      // re-apply highlight
      const hlPoint = stats[currentEpisode] ? [{{ x: currentEpisode + 1, y: stats[currentEpisode][def.key] }}] : [];
      chart.data.datasets[1].data = hlPoint;
      chart.update('none');
    }});
  }}

  function updateChartHighlight(ep) {{
    CHART_DEFS.forEach(def => {{
      const chart = charts[def.key];
      if (!chart || !episodeStats[ep]) return;
      chart.data.datasets[1].data = [{{ x: ep + 1, y: episodeStats[ep][def.key] }}];
      chart.update('none');
    }});
  }}

  // ── Polling ───────────────────────────────────────────────────────────────

  async function fetchEpisode(epIdx, since=0) {{
    try {{
      const r = await fetch(`${{MANAGER_URL}}/sim/${{SESSION_ID}}/frames/${{epIdx}}?since=${{since}}`);
      if (r.status === 404) return null;
      if (!r.ok) return null;
      return await r.json();
    }} catch (e) {{ return null; }}
  }}

  async function fetchEpisodeStats() {{
    try {{
      const r = await fetch(`${{MANAGER_URL}}/sim/${{SESSION_ID}}/episode_stats`);
      if (!r.ok) return null;
      return await r.json();
    }} catch (e) {{ return null; }}
  }}

  async function pollStatus() {{
    try {{
      const r = await fetch(`${{MANAGER_URL}}/sim/${{SESSION_ID}}/status`);
      if (!r.ok) return null;
      return await r.json();
    }} catch (e) {{ return null; }}
  }}

  async function tick() {{
    const [status, statsResult] = await Promise.all([pollStatus(), fetchEpisodeStats()]);

    if (!status) {{
      statusBar.textContent = '⏳ Waiting for simulation container to start...';
      statusBar.className = '';
      setTimeout(tick, POLL_INTERVAL_MS);
      return;
    }}

    const completed = status.completed_episodes || 0;
    simDone = status.done || false;
    loadingBar.style.width = (simDone ? 100 : Math.round(completed / NUM_EPISODES * 100)) + '%';

    // Update stats charts
    if (statsResult && statsResult.episode_stats && statsResult.episode_stats.length > 0) {{
      episodeStats = statsResult.episode_stats;
      updateCharts(episodeStats);
    }}

    // Stream frames for each known episode (including the in-progress one).
    // A given episode may need multiple chunks; we advance fetchedEpisodes
    // only when the server confirms the episode is complete.
    let epIdx = fetchedEpisodes;
    while (epIdx < completed) {{
      const offset = epFrameOffsets[epIdx] || 0;
      const data = await fetchEpisode(epIdx, offset);
      if (data === null) break;
      if (data.frames && data.frames.length > 0) {{
        const isContinuation = offset > 0;
        const boundary = isContinuation ? boundaries[epIdx] : frames.length;
        appendFrames(data.frames, epIdx, boundary, isContinuation);
        epFrameOffsets[epIdx] = data.next_since;
      }}
      if (data.is_complete) {{
        fetchedEpisodes = epIdx + 1;
        epIdx++;
      }} else {{
        break;  // mid-episode; wait for next tick to grab more frames
      }}
    }}

    if (simDone && fetchedEpisodes >= completed) {{
      statusBar.textContent = `✅ Done — ${{frames.length}} frames across ${{boundaries.length}} episode(s)`;
      statusBar.className = 'done';
      loadingBar.style.width = '100%';
      fetch(`${{MANAGER_URL}}/session/${{SESSION_ID}}`, {{ method: 'DELETE' }}).catch(() => {{}});
      return;
    }} else {{
      const ep = (status.stats && status.stats.episode !== undefined) ? status.stats.episode : fetchedEpisodes;
      statusBar.textContent = `⏳ Episode ${{ep + 1}} / ${{NUM_EPISODES}} running — ${{frames.length}} frames loaded`;
      statusBar.className = '';
    }}

    setTimeout(tick, POLL_INTERVAL_MS);
  }}

  // ── Init ─────────────────────────────────────────────────────────────────
  img.onload = sendHeight;
  function sendHeight() {{
    const h = document.body.scrollHeight + 8;
    window.parent.postMessage({{type: 'streamlit:setFrameHeight', height: h}}, '*');
  }}
  new ResizeObserver(sendHeight).observe(document.body);

  initCharts();
  tick();
</script>
</body>
</html>
"""
    components.html(html, height=1800, scrolling=False)


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
    session = create_session(num_episodes, size, n_agvs, n_pickers)
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
