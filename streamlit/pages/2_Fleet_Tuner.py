"""Fleet Sizing Tuner — sweep (num_agvs, num_pickers) combos and report
the Pareto-optimal trade-off between throughput (bin or pick) and total
fleet size.

Resource-gated: requires `TUNER_ENABLED=1` AND a private/local IP. See
streamlit/access.py.

Concurrency model: parallel within each picker row. Multiple AGV combos
for the same picker count run simultaneously (speculative execution),
with results processed in AGV-ascending order so plateau detection still
observes them in sequence. When a row plateaus, in-flight speculative
sessions past the plateau point are cancelled and those slots immediately
flow to other active rows. A 5950X-class host can comfortably sustain
~12 concurrent headless sims.
"""

from __future__ import annotations

import io
import os
import time
from collections import deque
from statistics import mean

import altair as alt
import pandas as pd
import requests
import streamlit as st

from access import tuner_access_state


st.set_page_config(page_title="Fleet Tuner", page_icon="🔬", layout="wide")

MANAGER_URL = os.environ.get("MANAGER_URL", "http://manager:8001")
POLL_INTERVAL_SECONDS = 2

METRIC_LABELS = {
    "bin_throughput":  "Bins delivered / hr (AGV side)",
    "pick_throughput": "Items picked / hr (picker side)",
}

# ── Access gate ───────────────────────────────────────────────────────────────

enabled, reason = tuner_access_state()

st.title("🔬 Fleet Sizing Tuner")
st.caption(
    "Sweep (AGVs × Pickers) for a chosen map and report Pareto-optimal "
    "fleet sizes. Each combo runs a headless sim container (no frame "
    "capture). Multiple combos run in parallel, up to the concurrency cap."
)

if not enabled:
    st.warning(reason, icon="🔒")
    st.caption(
        "Controls below are visible but disabled. The fleet tuner is "
        "a local-only feature — see `docs/fleet-tuner.md` for setup."
    )

# ── Session state init ────────────────────────────────────────────────────────

defaults = {
    "tuner_running": False,
    # picker_count -> {
    #   "agv_queue":      deque[int],         # not-yet-spawned, ascending
    #   "in_flight":      dict[int, str],     # agv -> session_id (parallel)
    #   "results_buffer": dict[int, dict],    # arrived, awaiting in-order check
    #   "next_to_check":  int,                # next agv count to plateau-check
    #   "prev_best":      float,
    #   "plateau_counter":int,
    #   "done":           bool,
    # }
    "tuner_rows": {},
    "tuner_results": [],
    # session_id -> {"picker": int, "agvs": int, "started_at": float}
    "tuner_active_sessions": {},
    "tuner_stop": False,
    "tuner_config": None,
    "tuner_total_planned": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Stop flag handler — runs at the top of every rerun.
if st.session_state.tuner_stop:
    for sid in list(st.session_state.tuner_active_sessions):
        try:
            requests.delete(f"{MANAGER_URL}/session/{sid}", timeout=5)
        except Exception:
            pass
    st.session_state.tuner_active_sessions = {}
    st.session_state.tuner_running = False
    st.session_state.tuner_rows = {}
    st.session_state.tuner_stop = False


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_maps() -> list[str]:
    try:
        r = requests.get(f"{MANAGER_URL}/maps", timeout=5)
        return r.json().get("maps", []) or []
    except Exception:
        return []


def spawn_session(map_name: str, num_agvs: int, num_pickers: int,
                  episodes: int) -> str | None:
    """Create a headless sim session and return its session_id, or None on
    transient failure (the caller can retry on the next tick)."""
    payload = {
        "map_name": map_name,
        "num_agvs": int(num_agvs),
        "num_pickers": int(num_pickers),
        "num_episodes": int(episodes),
        "headless": True,
    }
    try:
        r = requests.post(f"{MANAGER_URL}/session", json=payload, timeout=15)
        if r.status_code == 503 and r.json().get("detail") == "building":
            # Image is being built — back off; another rerun will retry.
            return None
        r.raise_for_status()
        return r.json()["session_id"]
    except Exception:
        return None


def collect_session_result(session_id: str, meta: dict) -> dict:
    """Pull KPIs for a finished session and synthesize a results row.
    Always cleans up the session container."""
    result = {
        "num_agvs": meta["agvs"],
        "num_pickers": meta["picker"],
        "total_agents": meta["agvs"] + meta["picker"],
        "episodes_completed": 0,
        "bin_throughput": 0.0,
        "pick_throughput": 0.0,
        "agv_utilization": 0.0,
        "clash_rate": 0.0,
        "distance_per_bot": 0.0,
        "deliveries": 0.0,
        "items_picked": 0.0,
        "stucks": 0.0,
        "error": "",
    }
    try:
        ep_data = requests.get(
            f"{MANAGER_URL}/sim/{session_id}/episode_stats", timeout=5
        ).json()
        ep_stats = ep_data.get("episode_stats", []) or []
        if ep_stats:
            result.update({
                "episodes_completed": len(ep_stats),
                "bin_throughput":  mean(s.get("bin_throughput", 0) for s in ep_stats),
                "pick_throughput": mean(s.get("pick_throughput", 0) for s in ep_stats),
                "agv_utilization": mean(s.get("agv_utilization", 0) for s in ep_stats),
                "clash_rate": mean(s.get("clash_rate", 0) for s in ep_stats),
                "distance_per_bot": mean(s.get("distance_per_bot", 0) for s in ep_stats),
                "deliveries": mean(s.get("deliveries", 0) for s in ep_stats),
                "items_picked": mean(s.get("items_picked", 0) for s in ep_stats),
                "stucks": mean(s.get("stucks", 0) for s in ep_stats),
            })
        else:
            result["error"] = "no_episode_stats"
    except Exception as exc:
        result["error"] = f"fetch_failed: {str(exc)[:80]}"
    finally:
        try:
            requests.delete(f"{MANAGER_URL}/session/{session_id}", timeout=5)
        except Exception:
            pass
    return result


def pareto_optimal(rows: list[dict], metric: str) -> list[bool]:
    """A combo is Pareto-optimal if no other combo has BOTH a >= value on
    `metric` AND a <= total_agents, with at least one strict inequality."""
    flags = []
    for i, r in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if (other[metric] >= r[metric]
                    and other["total_agents"] <= r["total_agents"]
                    and (other[metric] > r[metric]
                         or other["total_agents"] < r["total_agents"])):
                dominated = True
                break
        flags.append(not dominated)
    return flags


# ── Sidebar: inputs ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Tuner Controls")

    available_maps = get_maps() or ["tiny_dhl"]
    map_name = st.selectbox("Warehouse Map", available_maps, disabled=not enabled)

    metric = st.selectbox(
        "Optimization metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda m: METRIC_LABELS[m],
        disabled=not enabled,
        help=("Drives plateau detection AND the Pareto frontier. "
              "`pick_throughput` requires Pickers (min) > 0 — items_picked "
              "is always 0 without pickers."),
    )

    agv_min = st.number_input("AGVs (min)", min_value=1, max_value=200, value=1,
                              disabled=not enabled)
    agv_max = st.number_input("AGVs (max)", min_value=1, max_value=200, value=20,
                              disabled=not enabled)
    picker_min = st.number_input("Pickers (min)", min_value=0, max_value=200, value=0,
                                 disabled=not enabled)
    picker_max = st.number_input("Pickers (max)", min_value=0, max_value=200, value=10,
                                 disabled=not enabled)
    episodes_per_combo = st.number_input(
        "Episodes per combo (seeds)", min_value=1, max_value=10, value=3,
        disabled=not enabled,
        help="More episodes per combo = lower variance, longer runtime.",
    )
    concurrency = st.number_input(
        "Concurrent sims", min_value=1, max_value=32, value=12,
        disabled=not enabled,
        help="Max simultaneous sim containers. Each uses ~1 CPU core "
             "(headless skips rendering). Rule of thumb: physical_cores − 2 "
             "to leave headroom for the manager and streamlit.",
    )
    plateau_tolerance_pct = st.number_input(
        "Plateau tolerance (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5,
        disabled=not enabled,
        help="Metric gain below this %, relative to current best in the row, "
             "counts as a plateau hit. Uses the chosen optimization metric.",
    )
    plateau_streak = st.number_input(
        "Plateau streak", min_value=1, max_value=5, value=2,
        disabled=not enabled,
        help="Consecutive plateau hits before stopping the AGV-axis sweep "
             "for that picker count.",
    )

    st.divider()
    start_clicked = st.button(
        "▶ Start Tuner",
        type="primary", use_container_width=True,
        disabled=not enabled or st.session_state.tuner_running,
    )
    if st.session_state.tuner_running:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.tuner_stop = True
            st.rerun()


# ── Start handler ─────────────────────────────────────────────────────────────

if start_clicked:
    if agv_max < agv_min:
        st.error("AGVs max must be >= AGVs min.")
    elif picker_max < picker_min:
        st.error("Pickers max must be >= Pickers min.")
    elif metric == "pick_throughput" and picker_min == 0:
        st.error(
            "`pick_throughput` requires Pickers (min) ≥ 1. "
            "With 0 pickers, items_picked is always 0 and the Pareto "
            "frontier degenerates. Either raise Pickers (min), or switch "
            "the optimization metric to `bin_throughput`."
        )
    else:
        rows = {}
        total = 0
        for p in range(int(picker_min), int(picker_max) + 1):
            agv_queue = deque(range(int(agv_min), int(agv_max) + 1))
            rows[p] = {
                "agv_queue": agv_queue,
                "in_flight": {},
                "results_buffer": {},
                "next_to_check": int(agv_min),
                "prev_best": float("-inf"),
                "plateau_counter": 0,
                "done": False,
            }
            total += len(agv_queue)

        st.session_state.tuner_rows = rows
        st.session_state.tuner_results = []
        st.session_state.tuner_active_sessions = {}
        st.session_state.tuner_running = True
        st.session_state.tuner_total_planned = total
        st.session_state.tuner_config = {
            "map_name": map_name,
            "metric": metric,
            "episodes_per_combo": int(episodes_per_combo),
            "plateau_tolerance": float(plateau_tolerance_pct) / 100.0,
            "plateau_streak": int(plateau_streak),
            "concurrency": int(concurrency),
        }
        st.rerun()


# ── Main display ──────────────────────────────────────────────────────────────

cfg = st.session_state.tuner_config

status_box = st.empty()
progress_box = st.empty()
inflight_box = st.empty()
chart_col, table_col = st.columns([3, 2])
heatmap_box = chart_col.empty()
pareto_box = chart_col.empty()
table_box = table_col.empty()
download_box = st.empty()


def render_outputs():
    results = st.session_state.tuner_results
    if not results:
        heatmap_box.info("No results yet. Configure and click **Start Tuner**.")
        return

    metric = (cfg or {}).get("metric", "bin_throughput")
    metric_label = METRIC_LABELS[metric]

    df = pd.DataFrame(results)
    flags = pareto_optimal(results, metric)
    df["pareto_optimal"] = flags

    heatmap = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("num_agvs:O", title="AGVs"),
            y=alt.Y("num_pickers:O", title="Pickers", sort="-y"),
            color=alt.Color(f"{metric}:Q", title=metric_label,
                            scale=alt.Scale(scheme="viridis")),
            tooltip=["num_agvs", "num_pickers",
                     "bin_throughput", "pick_throughput",
                     "agv_utilization", "clash_rate",
                     "distance_per_bot", "pareto_optimal"],
        )
        .properties(height=320, title=f"{metric_label} heatmap")
    )
    heatmap_box.altair_chart(heatmap, use_container_width=True)

    base = alt.Chart(df).encode(
        x=alt.X("total_agents:Q", title="Total agents (AGVs + Pickers)"),
        y=alt.Y(f"{metric}:Q", title=metric_label),
        tooltip=["num_agvs", "num_pickers",
                 "bin_throughput", "pick_throughput",
                 "agv_utilization", "pareto_optimal"],
    )
    pareto_layer = (
        base.transform_filter(alt.datum.pareto_optimal)
        .mark_point(size=140, filled=True, color="#d62728")
    )
    other_layer = (
        base.transform_filter("!datum.pareto_optimal")
        .mark_point(size=70, filled=True, color="#7f7f7f", opacity=0.6)
    )
    pareto_chart = alt.layer(other_layer, pareto_layer).properties(
        height=320,
        title=f"Pareto frontier — {metric_label} vs total agents (red = non-dominated)",
    )
    pareto_box.altair_chart(pareto_chart, use_container_width=True)

    display_df = df.sort_values(metric, ascending=False)[
        ["num_agvs", "num_pickers", "total_agents",
         "bin_throughput", "pick_throughput",
         "agv_utilization", "clash_rate",
         "distance_per_bot", "deliveries", "items_picked",
         "stucks", "episodes_completed", "pareto_optimal"]
    ]
    table_box.dataframe(display_df, use_container_width=True, height=420)

    buf = io.StringIO()
    display_df.to_csv(buf, index=False)
    download_box.download_button(
        "💾 Download results as CSV",
        data=buf.getvalue(),
        file_name=f"fleet_tuner_{cfg['map_name'] if cfg else 'unknown'}_{metric}.csv",
        mime="text/csv",
    )


# Status / progress
if st.session_state.tuner_running and cfg:
    completed = len(st.session_state.tuner_results)
    in_flight = len(st.session_state.tuner_active_sessions)
    total = st.session_state.tuner_total_planned
    pruned = total - completed - in_flight - sum(
        len(r["agv_queue"]) for r in st.session_state.tuner_rows.values()
    )
    progress_box.progress(
        completed / max(total, 1),
        text=f"Completed {completed}/{total} combos "
             f"(in flight: {in_flight}, pruned: {pruned})",
    )
elif cfg and not st.session_state.tuner_running and st.session_state.tuner_results:
    status_box.success(
        f"Tuner finished. {len(st.session_state.tuner_results)} combos run "
        f"out of {st.session_state.tuner_total_planned} planned "
        f"(rest pruned by plateau detection)."
    )

render_outputs()


# ── Concurrent run loop: poll, process finished, spawn new ────────────────────

def _all_rows_done() -> bool:
    return all(
        r["done"] and not r["in_flight"]
        for r in st.session_state.tuner_rows.values()
    )


if st.session_state.tuner_running and cfg:
    active = st.session_state.tuner_active_sessions
    rows_state = st.session_state.tuner_rows

    # 1. Poll all in-flight sessions; collect any that finished.
    finished_ids = []
    for sid in list(active):
        try:
            status = requests.get(
                f"{MANAGER_URL}/sim/{sid}/status", timeout=5
            ).json()
            if status.get("done"):
                finished_ids.append(sid)
        except Exception:
            # Sim not yet ready or transient error — try again next tick.
            pass

    # 2. For each finished session: stash result keyed by AGV count in the
    #    row's results_buffer. Append to the global results so the heatmap
    #    sees the data point even if it later gets pruned past plateau.
    for sid in finished_ids:
        meta = active.pop(sid)
        result = collect_session_result(sid, meta)
        st.session_state.tuner_results.append(result)
        row = rows_state[meta["picker"]]
        row["in_flight"].pop(meta["agvs"], None)
        # Buffer every result (including errors) so the plateau loop
        # can advance next_to_check past failed combos rather than
        # hanging the row indefinitely.
        row["results_buffer"][meta["agvs"]] = result

    # 3. Plateau-check IN ORDER, per row. Drain the buffer as long as
    #    next_to_check is present so plateau detection observes results
    #    in AGV-ascending order regardless of arrival order.
    for picker, row in rows_state.items():
        if row["done"]:
            continue
        while row["next_to_check"] in row["results_buffer"]:
            result = row["results_buffer"].pop(row["next_to_check"])
            row["next_to_check"] += 1
            if result.get("error"):
                # Errored combo: skip plateau accounting so a single
                # crashed sim doesn't trigger or break the streak.
                continue
            metric_value = result[cfg["metric"]]
            threshold = row["prev_best"] * (1.0 + cfg["plateau_tolerance"])
            if metric_value <= threshold:
                row["plateau_counter"] += 1
            else:
                row["plateau_counter"] = 0
            if metric_value > row["prev_best"]:
                row["prev_best"] = metric_value
            if row["plateau_counter"] >= cfg["plateau_streak"]:
                # Cancel speculative sessions past the plateau point so
                # their slots flow to other active rows immediately.
                for agv, sid in list(row["in_flight"].items()):
                    if agv >= row["next_to_check"]:
                        try:
                            requests.delete(
                                f"{MANAGER_URL}/session/{sid}", timeout=5
                            )
                        except Exception:
                            pass
                        row["in_flight"].pop(agv, None)
                        active.pop(sid, None)
                row["agv_queue"] = deque()
                row["results_buffer"] = {}
                row["done"] = True
                break

        if not row["agv_queue"] and not row["in_flight"] and not row["results_buffer"]:
            row["done"] = True

    # 4. Spawn: round-robin across active rows until cap or no work left.
    cap = cfg["concurrency"]
    made_progress = True
    while len(active) < cap and made_progress:
        made_progress = False
        # Ascending picker order each pass — lower picker counts get
        # priority but every row gets a turn within a pass.
        for picker in sorted(rows_state):
            if len(active) >= cap:
                break
            row = rows_state[picker]
            if row["done"] or not row["agv_queue"]:
                continue
            agv = row["agv_queue"].popleft()
            sid = spawn_session(
                cfg["map_name"], agv, picker, cfg["episodes_per_combo"]
            )
            if sid is None:
                # Manager refused (e.g. building) — re-queue and retry next tick.
                row["agv_queue"].appendleft(agv)
                continue
            row["in_flight"][agv] = sid
            active[sid] = {
                "picker": picker,
                "agvs": agv,
                "started_at": time.time(),
            }
            made_progress = True

    # 5. Status: list in-flight combos compactly.
    if active:
        flying = sorted((m["picker"], m["agvs"]) for m in active.values())
        inflight_box.info(
            "🚀 In flight: "
            + ", ".join(f"({a}A·{p}P)" for p, a in flying)
        )
    else:
        inflight_box.empty()

    # 6. Termination check.
    if _all_rows_done() and not active:
        st.session_state.tuner_running = False
        st.rerun()
    else:
        time.sleep(POLL_INTERVAL_SECONDS)
        st.rerun()
