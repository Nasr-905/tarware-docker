"""
Manager Service
---------------
  POST /session                       → create sim container
  POST /session/{id}/done             → called by sim when done (marks done, does NOT destroy)
  DELETE /session/{id}                → user-initiated teardown
  GET  /session/{id}/status           → session status
  GET  /sessions                      → list sessions
  GET  /health                        → health check
  GET  /sim/{id}/status               → proxy to sim /status  (browser-accessible)
  GET  /sim/{id}/frames/{episode}     → proxy to sim /frames/{episode} (browser-accessible)
"""

import os
import uuid
import time
import threading
from pathlib import Path
from typing import Dict, Optional

import docker
import docker.errors
import requests as req_lib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(root_path="/manager")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

docker_client = docker.from_env()

NETWORK_NAME = os.environ.get("DOCKER_NETWORK", "tarware_net")
SIM_IMAGE = os.environ.get("SIM_IMAGE", "tarware-simulation:latest")
SIM_CONTAINER_PREFIX = "tarware_sim_"

# TA-RWARE env vars forwarded to every spawned sim container
_map_name = os.environ.get("TARWARE_MAP_NAME", "tiny")
_TARWARE_SIM_ENV = {
    k: os.environ[k]
    for k in (
        "TARWARE_MAP_NAME",
        "TARWARE_AGVS",
        "TARWARE_PICKERS",
        "TARWARE_OBS_TYPE",
        "TARWARE_RENDER_TILE_SIZE",
        "TARWARE_REWARD_TYPE",
        "TARWARE_ORDER_CSV_PATH",
        "TARWARE_ABC_CSV_PATH",
        "TARWARE_STEPS_PER_SIMULATED_SECOND",
        "TARWARE_REQUEST_QUEUE_SIZE",
        "TARWARE_MAP_JSON_PATH",
        "TARWARE_RENDER_WIDTH",
        "TARWARE_RENDER_HEIGHT",
        "TARWARE_MAX_STEPS",
        "TARWARE_MAX_INACTIVITY_STEPS",
        "TARWARE_PICK_BASE_TICKS",
        "TARWARE_AGV_TRANSFER_BASE_SECONDS",
        "TARWARE_PICKER_MODEL",
        "TARWARE_ABSTRACT_PICKER_UPH",
        "FRAME_SCALE",
    )
    if k in os.environ
}
# Always compute MAP_CSV_PATH and MAP_JSON_PATH from the map name so they stay in sync
_TARWARE_SIM_ENV["TARWARE_MAP_CSV_PATH"] = (
    os.environ.get("TARWARE_MAP_CSV_PATH")
    or f"/app/tarware/data/maps/{_map_name}.csv"
)
_TARWARE_SIM_ENV["TARWARE_MAP_JSON_PATH"] = (
    os.environ.get("TARWARE_MAP_JSON_PATH")
    or f"/app/tarware/data/maps/{_map_name}.json"
)

sessions: Dict[str, dict] = {}
sessions_lock = threading.Lock()

# ── Image build state ─────────────────────────────────────────────────────────
_image_build_lock = threading.Lock()   # only one build at a time
_image_building = False                # true while build is in progress

# ── Orphan container GC ───────────────────────────────────────────────────────

ORPHAN_TTL_SECONDS = int(os.environ.get("ORPHAN_TTL_SECONDS", str(30 * 60)))  # 30 min default
GC_INTERVAL_SECONDS = int(os.environ.get("GC_INTERVAL_SECONDS", str(5 * 60)))  # check every 5 min

def _gc_loop():
    while True:
        time.sleep(GC_INTERVAL_SECONDS)
        now = time.time()

        # 1. Reap tracked sessions that have exceeded the TTL.
        # Headless (tuner) sessions are exempt — they can run for hours and
        # the user cleans them up manually (tuner is local-only).
        with sessions_lock:
            stale = [
                sid for sid, s in sessions.items()
                if not s.get("exempt_from_gc")
                and now - s["created_at"] > ORPHAN_TTL_SECONDS
            ]
        for sid in stale:
            print(f"[manager] GC: removing stale session {sid} (>{ORPHAN_TTL_SECONDS}s old)", flush=True)
            with sessions_lock:
                sessions.pop(sid, None)
            cleanup_container(sid)

        # 2. Reap any sim containers that aren't tracked (e.g. after a manager restart)
        try:
            with sessions_lock:
                tracked_names = {f"{SIM_CONTAINER_PREFIX}{sid}" for sid in sessions}
            for c in docker_client.containers.list(filters={"name": SIM_CONTAINER_PREFIX}):
                if c.name not in tracked_names:
                    print(f"[manager] GC: removing untracked container {c.name}", flush=True)
                    try:
                        c.stop(timeout=5)
                        c.remove(force=True)
                    except Exception as e:
                        print(f"[manager] GC: error removing {c.name}: {e}", flush=True)
        except Exception as e:
            print(f"[manager] GC scan error: {e}", flush=True)

threading.Thread(target=_gc_loop, daemon=True).start()


class SessionRequest(BaseModel):
    num_episodes: int = 10
    env_name: Optional[str] = None
    map_name: Optional[str] = None
    num_agvs: Optional[int] = None
    num_pickers: Optional[int] = None
    order_csv: Optional[str] = None  # filename only (e.g. "order_data_sample"); resolved against /app/tarware/data/processed/
    headless: bool = False  # skip frame rendering for KPI-only sweeps (fleet tuner)
    picker_model: Optional[str] = None  # "abstract" or "physical"
    abstract_picker_uph: Optional[float] = None


@app.get("/maps")
def list_maps():
    """List all available map names from the mounted simulation source.

    Consumers (simulator, tuner) decide their own UI exclusions.
    `full_dhl` is too large for the live-playback simulator but valid for
    the headless tuner."""
    maps_dir = Path("/sim-src/tarware/data/maps")
    try:
        maps = sorted(p.stem for p in maps_dir.glob("*.csv"))
    except Exception:
        maps = []
    return {"maps": maps}


@app.get("/order_datasets")
def list_order_datasets():
    """List available order CSVs from the mounted simulation source.

    Filters for filenames containing "order" so ABC analysis CSVs sitting
    in the same directory don't show up as order datasets.
    """
    processed_dir = Path("/sim-src/tarware/data/processed")
    try:
        datasets = sorted(
            p.stem for p in processed_dir.glob("*.csv")
            if "order" in p.stem.lower() and "abc" not in p.stem.lower()
        )
    except Exception:
        datasets = []
    return {"order_datasets": datasets}


def build_sim_image() -> bool:
    """Build the simulation image from the mounted build context. Blocks until done."""
    build_context = os.environ.get("SIM_BUILD_CONTEXT", "/sim-src")
    print(f"[manager] Building simulation image from {build_context!r}...", flush=True)
    if not os.path.isdir(build_context):
        print(f"[manager] ERROR: {build_context!r} is not a directory. Volume not mounted?", flush=True)
        print(f"[manager] / contents: {os.listdir('/')}", flush=True)
        return False
    if not os.path.exists(os.path.join(build_context, "Dockerfile")):
        print(f"[manager] ERROR: No Dockerfile found in {build_context!r}", flush=True)
        return False
    try:
        image, logs = docker_client.images.build(
            path=build_context,
            tag=SIM_IMAGE,
            rm=True,
        )
        for chunk in logs:
            if "stream" in chunk:
                line = chunk["stream"].rstrip()
                if line:
                    print(f"[manager] build: {line}", flush=True)
        print(f"[manager] Simulation image built successfully: {SIM_IMAGE}", flush=True)
        return True
    except Exception as e:
        print(f"[manager] Failed to build simulation image: {e}", flush=True)
        return False


def cleanup_container(session_id: str):
    container_name = f"{SIM_CONTAINER_PREFIX}{session_id}"
    try:
        container = docker_client.containers.get(container_name)
        container.stop(timeout=5)
        container.remove(force=True)
        print(f"[manager] Cleaned up container {container_name}", flush=True)
    except docker.errors.NotFound:
        pass
    except Exception as e:
        if "409" in str(e) or "removal of container" in str(e):
            pass
        else:
            print(f"[manager] Error cleaning up {container_name}: {e}", flush=True)


def get_sim_url(session_id: str) -> str:
    return f"http://{SIM_CONTAINER_PREFIX}{session_id}:8000"


# ── Browser-accessible proxy endpoints ───────────────────────────────────────

@app.get("/sim/{session_id}/status")
def proxy_sim_status(session_id: str):
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
    try:
        r = req_lib.get(f"{get_sim_url(session_id)}/status", timeout=5)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sim unreachable: {e}")


@app.get("/sim/{session_id}/episode_stats")
def proxy_sim_episode_stats(session_id: str):
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
    try:
        r = req_lib.get(f"{get_sim_url(session_id)}/episode_stats", timeout=10)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sim unreachable: {e}")


@app.get("/sim/{session_id}/frames/{episode}")
def proxy_sim_episode(session_id: str, episode: int):
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
    try:
        r = req_lib.get(f"{get_sim_url(session_id)}/frames/{episode}", timeout=60)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sim unreachable: {e}")


# ── Session lifecycle ─────────────────────────────────────────────────────────

@app.post("/session")
def create_session(req: SessionRequest):
    session_id = uuid.uuid4().hex[:8]
    container_name = f"{SIM_CONTAINER_PREFIX}{session_id}"

    # Clean up stale containers
    try:
        for c in docker_client.containers.list(all=True, filters={"name": SIM_CONTAINER_PREFIX}):
            if c.status in ("exited", "dead"):
                c.remove(force=True)
                print(f"[manager] Removed stale container {c.name}", flush=True)
    except Exception as e:
        print(f"[manager] Warning during stale cleanup: {e}", flush=True)

    session_env = dict(_TARWARE_SIM_ENV)
    if req.map_name:
        session_env["TARWARE_MAP_NAME"] = req.map_name
        session_env["TARWARE_MAP_CSV_PATH"] = f"/app/tarware/data/maps/{req.map_name}.csv"
        session_env["TARWARE_MAP_JSON_PATH"] = f"/app/tarware/data/maps/{req.map_name}.json"
    if req.num_agvs is not None:
        session_env["TARWARE_AGVS"] = str(req.num_agvs)
    if req.num_pickers is not None:
        session_env["TARWARE_PICKERS"] = str(req.num_pickers)
    if req.order_csv:
        session_env["TARWARE_ORDER_CSV_PATH"] = f"/app/tarware/data/processed/{req.order_csv}.csv"
    if req.headless:
        session_env["TARWARE_HEADLESS"] = "1"
    if req.picker_model:
        if req.picker_model not in ("abstract", "physical"):
            raise HTTPException(
                status_code=400,
                detail=f"picker_model must be 'abstract' or 'physical', got {req.picker_model!r}",
            )
        session_env["TARWARE_PICKER_MODEL"] = req.picker_model
    if req.abstract_picker_uph is not None:
        session_env["TARWARE_ABSTRACT_PICKER_UPH"] = str(req.abstract_picker_uph)

    try:
        container = docker_client.containers.run(
            image=SIM_IMAGE,
            name=container_name,
            detach=True,
            network=NETWORK_NAME,
            environment={
                **session_env,
                "DISPLAY": ":99",
                "MANAGER_URL": "http://manager:8001",
                "SESSION_ID": session_id,
            },
            auto_remove=False,
        )
    except docker.errors.ImageNotFound:
        global _image_building
        with _image_build_lock:
            if not _image_building:
                _image_building = True
                def _build_and_clear():
                    global _image_building
                    try:
                        build_sim_image()
                    finally:
                        _image_building = False
                threading.Thread(target=_build_and_clear, daemon=True).start()
        raise HTTPException(
            status_code=503,
            detail="building",
        )
    except Exception as e:
        import traceback
        print(f"[manager] ERROR: {e}\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to start container: {e}")

    sim_url = f"http://{container_name}:8000"

    with sessions_lock:
        sessions[session_id] = {
            "session_id": session_id,
            "container_name": container_name,
            "container_id": container.id,
            "env_name": str(_TARWARE_SIM_ENV),
            "num_episodes": req.num_episodes,
            "sim_url": sim_url,
            "created_at": time.time(),
            "status": "starting",
            # Headless sessions = fleet-tuner runs. They can take hours on
            # full_dhl and outlive the orphan TTL; the user manages cleanup
            # manually since the tuner is local-only.
            "exempt_from_gc": bool(req.headless),
        }

    def _start_sim():
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = req_lib.get(f"{sim_url}/health", timeout=2)
                if r.status_code == 200:
                    req_lib.post(f"{sim_url}/start", params={"num_episodes": req.num_episodes}, timeout=5)
                    with sessions_lock:
                        if session_id in sessions:
                            sessions[session_id]["status"] = "running"
                    return
            except Exception:
                pass
            time.sleep(2)
        with sessions_lock:
            if session_id in sessions:
                sessions[session_id]["status"] = "failed"

    threading.Thread(target=_start_sim, daemon=True).start()

    return {
        "session_id": session_id,
        "sim_url": sim_url,
        "status": "starting",
        "num_episodes": req.num_episodes,
    }


@app.post("/session/{session_id}/done")
def session_done(session_id: str):
    """Called by sim when all episodes finish. Marks done but does NOT destroy —
    the browser is still fetching frames via the proxy at this point."""
    with sessions_lock:
        if session_id not in sessions:
            return {"status": "already_gone"}
        sessions[session_id]["status"] = "done"
    return {"status": "done", "session_id": session_id}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """User-initiated or post-playback teardown."""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        del sessions[session_id]
    cleanup_container(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/session/{session_id}/status")
def get_session_status(session_id: str):
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = dict(sessions[session_id])
    try:
        container = docker_client.containers.get(session["container_name"])
        session["container_status"] = container.status
    except docker.errors.NotFound:
        session["container_status"] = "missing"
    return session


@app.get("/sessions")
def list_sessions():
    with sessions_lock:
        return {"sessions": list(sessions.values()), "count": len(sessions)}


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}
