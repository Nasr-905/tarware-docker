# KPI Tracking

How each KPI shown in the Streamlit dashboard is computed, where it comes from
in the simulation, and what conditions need to hold for the numbers to mean
what they're labelled as.

> **Verify before trusting.** Field names and code references in this doc are
> point-in-time (current as of the LogicalBin port). The "Conditions for
> accuracy" section captures non-derivable invariants and is the most durable
> part. The pipeline section will need re-checking against `simulation/server.py`
> and `simulation/tarware/tarware/warehouse.py` before relying on specifics.

---

## Pipeline

`simulation/server.py::simulation_loop` wraps the upstream
`tarware.heuristic.heuristic_episode()` and registers a `step_callback` that
fires after every `env.step()`. The callback accumulates per-step counters
out of the env's `info` dict, then per-episode stats are finalised when the
heuristic returns.

### Per-step accumulation (in the callback)

The env's `info["agvs_idle_time"]` and `info["agvs_distance_travelled"]` are
misnamed — they iterate `self.agents`, which contains AGVs **plus** pickers.
We don't use them. Instead, the callback derives AGV-only metrics directly
from per-agent state, slicing `env.agents[:env.num_agvs]` (agents are
stored AGVs-first):

```python
agvs = env_.agents[:env_.num_agvs]

# AGV-only idle: sum of AGVs whose req_action is NOOP or TOGGLE_LOAD
agv_idle = sum(
    1 for a in agvs
    if a.req_action in (Action.NOOP, Action.TOGGLE_LOAD)
)

# AGV-only distance: Manhattan delta from last-known position.
# Maintains a dict {agent.id: (x, y)} across callback invocations so
# we count actual displacement, not "FORWARD attempted".
agv_distance = 0
for a in agvs:
    prev = agv_prev_positions.get(a.id)
    if prev is not None:
        agv_distance += abs(a.x - prev[0]) + abs(a.y - prev[1])
    agv_prev_positions[a.id] = (a.x, a.y)

ep_state["deliveries"]   += info["shelf_deliveries"]   # event count, AGV-side
ep_state["items_picked"] += info["items_picked"]       # event count, picker-side
ep_state["agv_idle"]     += agv_idle                   # AGV-only
ep_state["clashes"]      += info["clashes"]            # AGV-only on env side
ep_state["stucks"]       += info["stucks"]             # event count
ep_state["distance"]     += agv_distance               # AGV-only
ep_state["timesteps"]    += 1
```

`agv_prev_positions` is a fresh dict per episode (the callback is
re-created inside the per-episode loop, and the env resets between
episodes), so position tracking doesn't leak across episodes.

### End-of-episode finalisation

| KPI                | Formula                                                         | Notes                                       |
|--------------------|-----------------------------------------------------------------|---------------------------------------------|
| `bin_throughput`   | `deliveries × 3600 ÷ (seconds_per_step × timesteps)`            | AGV-side: bins delivered per *simulated* hr |
| `pick_throughput`  | `items_picked × 3600 ÷ (seconds_per_step × timesteps)`          | picker-side: units picked per simulated hr  |
| `deliveries`       | sum of `info["shelf_deliveries"]`                               | total bin deliveries per episode            |
| `items_picked`     | sum of `info["items_picked"]`                                   | total units picked per episode              |
| `agv_utilization`  | `100 × (1 − agv_idle ÷ (n_agvs × timesteps))`                   | percent of agent-steps spent moving         |
| `clash_rate`       | `clashes ÷ timesteps × 100`                                     | percent of steps with a movement clash      |
| `distance_per_bot` | `distance ÷ n_agvs`                                             | total grid-cell moves per bot               |
| `stucks`           | sum of `info["stucks"]`                                         | total stuck-resolution events               |
| `fps`              | `timesteps ÷ wall_clock_elapsed`                                | wall-clock throughput, useful for debug     |
| `global_return`    | from `heuristic_episode` return value                           | total reward across all agents              |

> **Why two throughput metrics?** `bin_throughput` measures AGV-side work
> (shelves carried to the pickerwall). `pick_throughput` measures
> picker-side work (units physically removed from bins by pickers). With
> `num_pickers=0`, no pick events fire and `pick_throughput` is zero —
> the picker stage of the env simply never runs. Use `bin_throughput`
> when comparing AGV fleets only; use `pick_throughput` when end-to-end
> picker output is what you care about. The old `pick_rate` field was
> misnamed (it measured deliveries, not picks) and has been removed.

`seconds_per_step = 1.0 / TARWARE_STEPS_PER_SIMULATED_SECOND` (read from env
var). With `TARWARE_STEPS_PER_SIMULATED_SECOND=0.25`, each step represents
4 simulated seconds.

`n_agvs = len(env_raw.agents)` — see caveat below.

### Storage and exposure

- Per-episode stats go into `all_episode_stats` (a list, one entry per episode).
- `sim_stats` always reflects the **latest** completed episode (overwritten each iteration).
- The Streamlit dashboard plots `all_episode_stats` directly — one bar per episode, no averaging.
- The `/episode_stats` endpoint serves the full list; `/status` exposes `sim_stats`.

---

## Conditions for KPIs to be accurate

### 1. The episode must run to completion

The accumulators only finalise *after* `heuristic_episode()` returns. If the
container is killed mid-episode (orphan GC at 30 min, manual `docker rm`,
OOM, container crash), no `ep_stat` row is appended for that episode and
nothing partial is reported. Lost runs are simply absent.

### 2. `TARWARE_STEPS_PER_SIMULATED_SECOND` consistency

`server.py` reads this env var directly and uses it in the `bin_throughput`
and `pick_throughput` denominators. The env reads the same env var in
`tarware/__init__.py` and threads it into the warehouse. They will match
unless someone overrides one without the other.

> **Safer derivation:** read `info["steps_per_simulated_second"]` (the env
> emits this every step) instead of the env var. Future-proof against
> mismatches. Currently we use the env var for simplicity.

### 3. Single-episode interpretation

`sim_stats` only holds the **last** episode. `all_episode_stats` holds all
of them. The dashboard plots per-episode bars and does **not** average across
episodes. If you want a "mean over N runs" KPI, that is a Streamlit-side
computation, not a server-side guarantee.

### 4. First-frame discard

Each episode begins with a throwaway `env_raw.render(mode="rgb_array")` to
prime the pyglet/Xvfb viewer (the first render of a fresh viewer is clipped).
This render is read-only and does not touch any KPI counter.

### 5. Frame-capture lock

The `sim_lock` only guards the shared `all_frames` buffer. KPI state lives in
a per-callback closure (`ep_state`), no cross-thread mutation. The lock has
no effect on KPI accuracy.

### 6. Multi-episode runs use sequential seeds

`heuristic_episode(env, seed=episode, ...)` is called per iteration with
`seed=0, 1, 2, …`. Different seeds produce different order-arrival patterns,
so per-episode KPIs aren't directly comparable as repeated trials of the
same scenario — they're trials of *different* scenarios drawn from the same
distribution. For repeated-trial averaging, use the same seed each episode.

### 7. The heuristic's `request_queue` priority is deterministic

The reference heuristic sorts the request queue by `(-sku_backlog,
-on_replenishment, item.id)` (heuristic.py:339-345). Same seed + same map +
same heuristic = same KPIs. There is no randomness in the dispatch policy
itself.

---

## Source field reference

Every per-step counter the env emits, from `warehouse.py::_build_info`
(lines 3163-3189):

| info key                            | What it is                                          | Filtered to AGVs?   |
|-------------------------------------|-----------------------------------------------------|---------------------|
| `shelf_deliveries`                  | bins delivered to pickerwall this step              | n/a (event count, AGV-side)  |
| `items_picked`                      | units physically picked from bins this step (sum of `claim.sku_entry.quantity` for each completed `PickerTask` claim) | n/a (event count, picker-side) |
| `clashes`                           | movement conflicts this step                        | yes (AGV-only)      |
| `stucks`                            | stuck-resolution events this step                   | unclear — verify    |
| `picker_yields`                     | picker yield events this step                       | picker-only         |
| `agvs_idle_time`                    | count of agents with req_action ∈ {NOOP, TOGGLE_LOAD} this step | **NO — all agents** |
| `agvs_distance_travelled`           | total grid-cell moves this step                     | **NO — all agents** |
| `vehicles_busy`                     | per-agent `busy` flag, length = num_agvs + num_pickers | per-agent array  |
| `steps_per_simulated_second`        | env's actual step→time conversion                   | n/a                 |
| `simulated_seconds`                 | cumulative simulated time at this step              | n/a                 |
| `real_seconds`                      | cumulative real time at this step                   | n/a                 |
| `agv_nominal_cells_per_step`        | AGV speed config                                    | yes                 |
| `picker_nominal_cells_per_step`     | picker speed config                                 | yes (picker-only)   |
| `motion_speed_model`                | "physical_m_s" or "cells_per_step"                  | n/a                 |
| `agv_cells_per_step_configured`     | configured AGV speed                                | yes                 |
| `agv_cells_per_step_effective`      | post-clamp AGV speed                                | yes                 |
| `picker_cells_per_step_configured`  | configured picker speed                             | yes (picker-only)   |

`vehicles_busy` is the most useful field for AGV-only derivations because
it's per-agent and ordered (AGVs first, then pickers).

---

## Why we don't trust `info["agvs_*"]`

The env's `_build_info` (`warehouse.py::_build_info`, lines 3163-3189)
emits `info["agvs_idle_time"]` and `info["agvs_distance_travelled"]` that
look like AGV-only counters but are actually computed across all agents:

```python
# warehouse.py:3173 — iterates self.agents (AGVs + pickers)
agvs_idle_time = sum(
    int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD))
    for agent in self.agents
)

# warehouse.py:977-1084 (attribute_macro_actions) — same
for agent, macro_action in zip(self.agents, macro_actions):
    ...
    if can_move:
        agvs_distance_travelled += 1
```

The misnaming dates from the legacy single-agent-type setup; the new
branch added pickers but didn't rename the counter. If you ever
re-enable `agvs_idle_time` / `agvs_distance_travelled` directly, AGV
utilization and distance get silently polluted by picker activity.

The callback in `simulation_loop` deliberately bypasses these and
derives AGV-only versions from `env.agents[:env.num_agvs]`. The env
itself remains unchanged — no submodule patch required.

---

## Files

- `simulation/server.py::simulation_loop` — KPI accumulation and finalisation
- `simulation/server.py::on_step` — per-step callback wired into heuristic
- `simulation/tarware/tarware/warehouse.py::_build_info` — env-side info dict
- `simulation/tarware/tarware/warehouse.py::attribute_macro_actions` — distance counter
- `simulation/tarware/tarware/heuristic.py::heuristic_episode` — outer loop driving step_callback
- `streamlit/app.py` — dashboard rendering of `all_episode_stats`
