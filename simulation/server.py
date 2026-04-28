import base64
import io
import os
import threading
import time

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

os.environ.setdefault("DISPLAY", ":99")
# Tile size for rendering — picked up by tarware.rendering.Viewer at init time
os.environ.setdefault("TARWARE_RENDER_TILE_SIZE", "60")

import gymnasium as gym
import tarware  # noqa: F401 — registers the env via env vars at import time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

all_frames: list = []          # stores every frame across all episodes
episode_boundaries: list = []  # frame index where each episode starts
all_episode_stats: list = []   # one dict per completed episode
sim_stats: dict = {}
sim_lock = threading.Lock()
sim_thread: threading.Thread | None = None
sim_done = threading.Event()


FRAME_SCALE = float(os.environ.get("FRAME_SCALE", "0.6"))
# When HEADLESS, skip env.render() + JPEG encode + frame buffering on every step.
# Used by the fleet tuner to run KPI-only sweeps without paying rendering cost.
HEADLESS = os.environ.get("TARWARE_HEADLESS", "0") == "1"

def frame_to_jpeg_b64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame.astype("uint8"), "RGB")
    if FRAME_SCALE != 1.0:
        w, h = img.size
        img = img.resize((int(w * FRAME_SCALE), int(h * FRAME_SCALE)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def simulation_loop(env_name: str, num_episodes: int):
    from tarware.heuristic import heuristic_episode
    from tarware.definitions import Action

    sim_done.clear()
    env = gym.make(env_name)
    env_raw = env.unwrapped

    steps_per_sim_second = float(os.environ.get("TARWARE_STEPS_PER_SIMULATED_SECOND", "1.0"))
    seconds_per_step = 1.0 / steps_per_sim_second if steps_per_sim_second > 0 else 1.0

    for episode in range(num_episodes):
        # Warm up the renderer before heuristic_episode's internal reset.
        # The first render of a new pyglet viewer is clipped under Xvfb timing.
        # Skip in headless mode so the pyglet viewer is never instantiated.
        env_raw.reset(seed=episode)
        if not HEADLESS:
            _ = env_raw.render(mode="rgb_array")

        with sim_lock:
            episode_boundaries.append(len(all_frames))

        ep_state = {
            "deliveries": 0,
            "items_picked": 0,
            "agv_idle": 0,
            "clashes": 0,
            "stucks": 0,
            "distance": 0,
            "timesteps": 0,
        }
        # Per-AGV last-known position so we can derive AGV-only distance from
        # actual displacement (env's `info["agvs_distance_travelled"]` is
        # misnamed — it actually counts moves across all agents including
        # pickers, so we can't trust it once num_pickers > 0).
        agv_prev_positions: dict = {}
        start = time.time()

        def on_step(env_, info, _state=ep_state, _prev=agv_prev_positions, **_kwargs):
            if not HEADLESS:
                frame = env_.render(mode="rgb_array")
                if frame is not None:
                    with sim_lock:
                        all_frames.append(frame_to_jpeg_b64(frame))

            # AGV-only metrics derived from per-agent state. Agents are
            # stored AGVs-first, then pickers, so slicing by num_agvs is
            # safe.
            n_agvs = env_.num_agvs
            agvs = env_.agents[:n_agvs]
            agv_idle = sum(
                1 for a in agvs
                if a.req_action in (Action.NOOP, Action.TOGGLE_LOAD)
            )
            agv_distance = 0
            for a in agvs:
                prev = _prev.get(a.id)
                if prev is not None:
                    agv_distance += abs(a.x - prev[0]) + abs(a.y - prev[1])
                _prev[a.id] = (a.x, a.y)

            _state["deliveries"]   += info.get("shelf_deliveries", 0)
            _state["items_picked"] += info.get("items_picked", 0)
            _state["agv_idle"]     += agv_idle
            _state["clashes"]      += info.get("clashes", 0)
            _state["stucks"]       += info.get("stucks", 0)
            _state["distance"]     += agv_distance
            _state["timesteps"]    += 1

        _, global_return, _ = heuristic_episode(env_raw, seed=episode, step_callback=on_step)

        elapsed = time.time() - start
        timestep = ep_state["timesteps"]
        fps = timestep / elapsed if elapsed > 0 else 0
        # Throughputs are per *simulated* hour. Each step represents
        # `seconds_per_step` simulated seconds.
        # bin_throughput  = AGV bins delivered to pickerwall.
        # pick_throughput = picker units physically removed from bins.
        bin_throughput = (
            ep_state["deliveries"] * 3600 / (seconds_per_step * timestep)
            if timestep > 0 else 0
        )
        pick_throughput = (
            ep_state["items_picked"] * 3600 / (seconds_per_step * timestep)
            if timestep > 0 else 0
        )

        n_agvs = env_raw.num_agvs
        agv_util = round(100 * (1 - ep_state["agv_idle"] / max(n_agvs * timestep, 1)), 1)
        clash_rate = round(ep_state["clashes"] / max(timestep, 1) * 100, 2)
        distance_per_bot = round(ep_state["distance"] / n_agvs, 1) if n_agvs > 0 else 0

        ep_stat = {
            "episode": episode,
            "bin_throughput": round(bin_throughput, 2),
            "pick_throughput": round(pick_throughput, 2),
            "global_return": round(float(global_return), 2),
            "deliveries": ep_state["deliveries"],
            "items_picked": ep_state["items_picked"],
            "fps": round(fps, 2),
            "timesteps": timestep,
            "agv_utilization": agv_util,
            "clash_rate": clash_rate,
            "stucks": ep_state["stucks"],
            "distance_per_bot": distance_per_bot,
        }

        with sim_lock:
            all_episode_stats.append(ep_stat)
            sim_stats.update({
                "episode": episode,
                "num_episodes": num_episodes,
                **ep_stat,
            })

    env.close()
    sim_done.set()

    manager_url = os.environ.get("MANAGER_URL", "")
    session_id = os.environ.get("SESSION_ID", "")
    if manager_url and session_id:
        import requests as req_lib
        try:
            req_lib.post(f"{manager_url}/session/{session_id}/done", timeout=5)
        except Exception as e:
            print(f"[sim] Could not notify manager: {e}")


@app.post("/start")
def start(num_episodes: int = 10):
    global sim_thread
    env_name = tarware.ENV_ID
    sim_thread = threading.Thread(
        target=simulation_loop, args=(env_name, num_episodes), daemon=True
    )
    sim_thread.start()
    return {"status": "started", "num_episodes": num_episodes, "env": env_name}


@app.get("/status")
def get_status():
    with sim_lock:
        return {
            "done": sim_done.is_set(),
            "total_frames": len(all_frames),
            "completed_episodes": len(episode_boundaries),
            "stats": dict(sim_stats),
            "episode_boundaries": list(episode_boundaries),
        }


@app.get("/frames/{episode}")
def get_episode_frames(episode: int, since: int = 0):
    """Return frames for an episode (0-indexed).

    With `?since=N`, returns only frames at offset N+ *within the episode*.
    Used by the player to stream in-progress episodes incrementally instead
    of waiting for the entire episode to complete.

    Response includes `next_since` so the client can pass it back on the
    next call, and `is_complete` so the client knows when to advance to
    the next episode (a later boundary has been appended OR sim_done).
    """
    with sim_lock:
        boundaries = list(episode_boundaries)
        total_frames = len(all_frames)
        done_flag = sim_done.is_set()

    if episode >= len(boundaries):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Episode not yet available")

    start = boundaries[episode]
    end = boundaries[episode + 1] if episode + 1 < len(boundaries) else total_frames
    is_complete = (episode + 1 < len(boundaries)) or done_flag

    abs_start = min(end, start + max(0, since))
    with sim_lock:
        frames = list(all_frames[abs_start:end])

    return {
        "episode": episode,
        "frames": frames,
        "start_frame": start,
        "since": since,
        "next_since": end - start,
        "is_complete": is_complete,
    }


@app.get("/frames")
def get_frames():
    """Return all collected frames as a JSON array of base64 JPEG strings."""
    with sim_lock:
        return {
            "frames": list(all_frames),
            "episode_boundaries": list(episode_boundaries),
            "stats": dict(sim_stats),
        }


@app.get("/stats")
def get_stats():
    with sim_lock:
        return dict(sim_stats)


@app.get("/episode_stats")
def get_episode_stats():
    with sim_lock:
        return {
            "episode_stats": list(all_episode_stats),
            "done": sim_done.is_set(),
        }


@app.get("/health")
def health():
    return {"status": "ok"}
