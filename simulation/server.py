import asyncio
import base64
import io
import json
import os
import threading
import time
from collections import deque

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

os.environ.setdefault("DISPLAY", ":99")

import gymnasium as gym
import tarware  # noqa: F401

# ── Monkey-patch Viewer for higher resolution ─────────────────────────────────
import tarware.rendering as _r

_GRID_SIZE = int(os.environ.get("GRID_SIZE", "60"))
_original_viewer_init = _r.Viewer.__init__

def _patched_viewer_init(self, world_size):
    _original_viewer_init(self, world_size)
    self.grid_size = _GRID_SIZE
    self.icon_size = _GRID_SIZE * 2 // 3
    self.width = 1 + self.cols * (self.grid_size + 1)
    self.height = 1 + self.rows * (self.grid_size + 1)
    self.window.set_size(self.width, self.height)

_r.Viewer.__init__ = _patched_viewer_init
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

all_frames: list = []          # stores every frame across all episodes
episode_boundaries: list = []  # frame index where each episode starts
sim_stats: dict = {}
sim_lock = threading.Lock()
sim_thread: threading.Thread | None = None
sim_done = threading.Event()


FRAME_SCALE = float(os.environ.get("FRAME_SCALE", "0.6"))

def frame_to_jpeg_b64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame.astype("uint8"), "RGB")
    if FRAME_SCALE != 1.0:
        w, h = img.size
        img = img.resize((int(w * FRAME_SCALE), int(h * FRAME_SCALE)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def simulation_loop(env_name: str, num_episodes: int):
    from collections import OrderedDict
    from tarware.heuristic import MissionType, Mission
    from tarware.utils.utils import flatten_list, split_list
    from tarware.warehouse import AgentType

    sim_done.clear()
    env = gym.make(env_name)
    episode = 0

    while episode < num_episodes:
        env_raw = env.unwrapped
        _ = env_raw.reset(seed=episode)

        non_goal_location_ids = []
        for id_, coords in env_raw.action_id_to_coords_map.items():
            if (coords[1], coords[0]) not in env_raw.goals:
                non_goal_location_ids.append(id_)
        non_goal_location_ids = np.array(non_goal_location_ids)
        location_map = env_raw.action_id_to_coords_map
        coords_original_loc_map = {v: k for k, v in env_raw.action_id_to_coords_map.items()}

        agents = env_raw.agents
        agvs = [a for a in agents if a.type == AgentType.AGV]
        pickers = [a for a in agents if a.type == AgentType.PICKER]
        sections = env_raw.rack_groups
        picker_sections = split_list(sections, len(pickers))
        picker_sections = [flatten_list(l) for l in picker_sections]

        assigned_agvs = OrderedDict()
        assigned_pickers = OrderedDict()
        assigned_items = OrderedDict()

        done = False
        timestep = 0
        ep_return = 0.0
        ep_deliveries = 0
        start = time.time()

        # Render and discard the first frame of each episode — it renders clipped
        # due to Xvfb/pyglet initialization timing
        _throwaway = env_raw.render(mode="rgb_array")

        with sim_lock:
            episode_boundaries.append(len(all_frames))

        while not done:
            request_queue = env_raw.request_queue
            goal_locations = env_raw.goals
            actions = {k: 0 for k in agents}

            for item in request_queue:
                if item.id in assigned_items.values():
                    continue
                available_agvs = [a for a in agvs if not a.busy and not a.carrying_shelf and a not in assigned_agvs]
                if not available_agvs:
                    continue
                agv_paths = [env_raw.find_path((a.y, a.x), (item.y, item.x), a, care_for_agents=False) for a in available_agvs]
                closest_agv = available_agvs[np.argmin([len(p) for p in agv_paths])]
                item_loc_id = coords_original_loc_map[(item.y, item.x)]
                assigned_agvs[closest_agv] = Mission(MissionType.PICKING, item_loc_id, item.x, item.y, timestep)
                assigned_items[closest_agv] = item.id

            for agv in agvs:
                if agv in assigned_agvs and (agv.x == assigned_agvs[agv].location_x) and (agv.y == assigned_agvs[agv].location_y):
                    assigned_agvs[agv].at_location = True
                if agv not in assigned_agvs or agv.busy:
                    continue
                m = assigned_agvs[agv]
                if m.mission_type == MissionType.PICKING and m.at_location and agv.carrying_shelf:
                    goal_paths = [env_raw.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (x, y) in goal_locations]
                    closest_goal = goal_locations[np.argmin([len(p) for p in goal_paths])]
                    goal_loc_id = coords_original_loc_map[(closest_goal[1], closest_goal[0])]
                    assigned_agvs[agv] = Mission(MissionType.DELIVERING, goal_loc_id, closest_goal[0], closest_goal[1], timestep)
                m = assigned_agvs[agv]
                if m.mission_type == MissionType.DELIVERING and m.at_location and agv.carrying_shelf:
                    empty_shelves = env_raw.get_empty_shelf_information()
                    empty_ids = [i for i in list(non_goal_location_ids[empty_shelves > 0])
                                 if i not in [ms.location_id for ms in assigned_agvs.values()]]
                    empty_yx = [location_map[i] for i in empty_ids]
                    empty_paths = [env_raw.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (y, x) in empty_yx]
                    closest_id = empty_ids[np.argmin([len(p) for p in empty_paths])]
                    closest_yx = location_map[closest_id]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, closest_id, closest_yx[1], closest_yx[0], timestep)
                m = assigned_agvs[agv]
                if m.mission_type == MissionType.RETURNING and m.at_location and not agv.carrying_shelf:
                    assigned_agvs.pop(agv)
                    assigned_items.pop(agv)

            for agv, mission in assigned_agvs.items():
                if mission.mission_type in [MissionType.PICKING, MissionType.RETURNING]:
                    in_zone = [(mission.location_y, mission.location_x) in p for p in picker_sections]
                    if True in in_zone:
                        relevant_picker = pickers[in_zone.index(True)]
                        if relevant_picker not in assigned_pickers:
                            assigned_pickers[relevant_picker] = Mission(MissionType.PICKING, mission.location_id, mission.location_x, mission.location_y, timestep)

            for picker in pickers:
                if picker in assigned_pickers and (picker.x == assigned_pickers[picker].location_x) and (picker.y == assigned_pickers[picker].location_y):
                    assigned_pickers[picker].at_location = True
                    assigned_pickers.pop(picker)

            for agv, mission in assigned_agvs.items():
                actions[agv] = mission.location_id if not agv.busy else 0
            for picker, mission in assigned_pickers.items():
                actions[picker] = mission.location_id

            frame = env_raw.render(mode="rgb_array")
            if frame is not None:
                with sim_lock:
                    all_frames.append(frame_to_jpeg_b64(frame))

            _, reward, terminated, truncated, info = env_raw.step(list(actions.values()))
            done = all(terminated) or all(truncated)
            ep_return += float(np.sum(reward))
            ep_deliveries += info.get("shelf_deliveries", 0)
            timestep += 1

        elapsed = time.time() - start
        fps = timestep / elapsed if elapsed > 0 else 0
        pick_rate = ep_deliveries * 3600 / (5 * timestep) if timestep > 0 else 0

        with sim_lock:
            sim_stats.update({
                "episode": episode,
                "num_episodes": num_episodes,
                "timesteps": timestep,
                "global_return": round(ep_return, 2),
                "deliveries": ep_deliveries,
                "pick_rate": round(pick_rate, 2),
                "fps": round(fps, 2),
            })

        episode += 1

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
    env_name = os.environ.get("TARWARE_ENV", "tarware-tiny-3agvs-2pickers-partialobs-v1")
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
def get_episode_frames(episode: int):
    """Return frames for a single completed episode (0-indexed)."""
    with sim_lock:
        boundaries = list(episode_boundaries)
        total_frames = len(all_frames)

    if episode >= len(boundaries):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Episode not yet available")

    start = boundaries[episode]
    end = boundaries[episode + 1] if episode + 1 < len(boundaries) else total_frames

    # If this is the last known episode and sim isn't done, only return it when
    # the next boundary (or sim_done) confirms it's fully recorded
    if episode + 1 >= len(boundaries) and not sim_done.is_set():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Episode not yet complete")

    with sim_lock:
        frames = list(all_frames[start:end])

    return {
        "episode": episode,
        "frames": frames,
        "start_frame": start,
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


@app.get("/health")
def health():
    return {"status": "ok"}

