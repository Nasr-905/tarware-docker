# Hyperparameters Affecting KPIs

Every knob — env var, hard-coded constant, map-derived value — that influences
the numbers reported in `bin_throughput`, `pick_throughput`, `agv_utilization`,
`clash_rate`, `distance_per_bot`, `deliveries`, `items_picked`, `stucks`, or
`global_return`.

> **Throughput-metric naming.** AGVs drive `bin_throughput` (bins delivered to
> the pickerwall per simulated hour); pickers drive `pick_throughput` (units
> physically picked per simulated hour). End-to-end picker output is
> bottlenecked by the *slower* stage. Use `bin_throughput` when comparing
> AGV fleets only; switch to `pick_throughput` when picker output is what you
> care about. (The legacy `pick_rate` field measured deliveries, not picks —
> it has been renamed to `bin_throughput`.)

> **Verify before trusting.** Defaults and key names are point-in-time. Re-grep
> `simulation/tarware/tarware/__init__.py`, `warehouse.py`, `heuristic.py`,
> `human_factors.py`, and `order_sequencer.py` before making decisions that
> depend on a specific number.

> **Companion doc.** See `docs/kpis.md` for how each KPI is computed and the
> conditions for accuracy. This file covers what *changes* them.

---

## How to read this doc

Each row gives:
- **Knob** — env var or hard-coded constant
- **Default** — current default in code
- **KPIs affected** — which dashboard KPIs change with this knob
- **Direction** — qualitative effect of increasing the value
- **Where set** — which file owns the default

Symbol legend:
- ↑ moves the KPI up
- ↓ moves the KPI down
- ↕ effect direction depends on map / load
- — no effect

---

## 1. Fleet sizing

The most direct levers on every KPI.

| Knob              | Default | `bin_throughput` | `pick_throughput` | `agv_utilization` | `clash_rate` | `distance_per_bot` | `deliveries` | Where        |
|-------------------|---------|------------------|-------------------|-------------------|--------------|---------------------|--------------|--------------|
| `TARWARE_AGVS`    | 3       | ↑                | ↑ (until pickers saturate) | ↓ (more AGVs → more idle if work-bound) | ↑ (more contention) | ↓ (work split) | ↑ (more throughput) | `__init__.py:42` |
| `TARWARE_PICKERS` | 4 (env), 0 (compose) | — (pickers don't deliver bins) | ↑ (until AGVs starve them) | — (AGV-only KPIs are picker-isolated) | ↑ (more agents on grid) | — | — | `__init__.py:44` |

**Notes**
- `agv_utilization` is non-monotonic: too few AGVs → utilization saturates near
  100% but throughput is low; too many → utilization drops because work doesn't
  fill them.
- With `num_pickers = 0`, no pick events fire — `pick_throughput` and
  `items_picked` are exactly zero. `bin_throughput` is unaffected and becomes
  the only meaningful throughput measure.
- Adding pickers does NOT increase `bin_throughput` (deliveries are an AGV-side
  event). It only increases `pick_throughput`. End-to-end pickerwall throughput
  is bottlenecked by whichever side is slower — use the Fleet Tuner's
  `pick_throughput` objective to size the picker fleet for picker-bound regimes.
- Increasing `TARWARE_PICKERS` only helps if the map *has* picker spawn cells
  (env clamps to `min(requested, len(picker_spawn_locs))`).

---

## 2. Time scaling

These change the simulated-vs-wall-clock relationship and so directly enter
both throughput KPIs (events per *simulated* hour).

| Knob                                | Default | KPIs affected                         | Direction                                      | Where              |
|-------------------------------------|---------|----------------------------------------|------------------------------------------------|--------------------|
| `TARWARE_STEPS_PER_SIMULATED_SECOND`| `1.0` (compose: `0.25`) | `bin_throughput` & `pick_throughput` (denominator), all rate KPIs | ↑ value → more steps/sim-sec → ↓ throughput (same events spread over more sim-time) | `__init__.py:50` |
| `TARWARE_MAX_STEPS`                 | `500`   | `deliveries`, `items_picked`, `distance_per_bot`, throughputs (sample size) | ↑ → episode runs longer → ↑ raw counts; throughputs stabilise | `__init__.py:57` |
| `TARWARE_MAX_INACTIVITY_STEPS`      | `None`  | All — episode terminates early on stall | If set: ↓ all cumulative KPIs when triggered | `__init__.py:53` |

> The compose default `TARWARE_STEPS_PER_SIMULATED_SECOND=0.25` makes 1 step ==
> 4 simulated seconds. Watch this when comparing runs with different settings —
> the throughput KPIs are normalised but raw step counts aren't.

### Physical-time / speed model

These only matter when `TARWARE_AGV_CELLS_PER_STEP` ≠ 1 or the physical-speed
model is enabled (motion credit applies).

| Knob                              | Default | KPIs affected                  | Direction              | Where                  |
|-----------------------------------|---------|---------------------------------|------------------------|------------------------|
| `TARWARE_AGV_CELLS_PER_STEP`      | `1.0`   | `distance_per_bot`, `bin_throughput` | ↑ → AGVs cover more ground per step → ↑ deliveries | `warehouse.py:418`    |
| `TARWARE_PICKER_CELLS_PER_STEP`   | `1.0`   | `pick_throughput` (if picker-bottlenecked), picker activity | ↑ → pickers consume bins faster | `warehouse.py:421`    |
| `TARWARE_AGV_NOMINAL_SPEED_M_S`   | `1.0`   | Same as `AGV_CELLS_PER_STEP` (physical model) | ↑ → faster AGVs | `human_factors.py:32` |
| `TARWARE_PICKER_NOMINAL_SPEED_M_S`| `1.0`   | `pick_throughput` | ↑ | `human_factors.py:33` |
| `TARWARE_GRID_CELL_SIZE_M`        | `1.0`   | Indirect — calibrates physical model | ↕ | `human_factors.py:31` |
| `TARWARE_REAL_SECONDS_PER_SIM_SECOND` | `1.0` | Wall-clock display only | — | `human_factors.py:29` |

---

## 3. Order arrival

Drives the `request_queue` that the heuristic consumes. No requests → no work →
all KPIs zero or trivial.

| Knob                          | Default                                    | KPIs affected            | Direction                                                  | Where               |
|-------------------------------|--------------------------------------------|---------------------------|------------------------------------------------------------|---------------------|
| `TARWARE_ORDER_CSV_PATH`      | `data/processed/order_data_sample.csv`     | All                       | Different order patterns → different bottlenecks           | `__init__.py:24`    |
| `TARWARE_REQUEST_QUEUE_SIZE`  | `20`                                       | `bin_throughput`, `agv_utilization` | ↓ → fewer concurrent orders → AGVs can starve; ↑ → near-saturation, AGVs consistently busy | `__init__.py:48`   |
| `TARWARE_ABC_CSV_PATH`        | `data/processed/abc_data_sample.csv`       | `distance_per_bot`, `bin_throughput` | A-class SKUs nearer pickerwall → ↓ travel → ↑ throughput | `__init__.py:33`    |

**Notes**
- Order timestamps in the CSV are normalised to step 0 (first order releases at
  start of episode). Order arrival shape *during* the episode follows the CSV.
- `REQUEST_QUEUE_SIZE` caps how many requests are visible to the heuristic at
  once. If raised much higher than `n_agvs × 3`, returns diminish (heuristic
  iterates the whole queue every step).

---

## 4. Map / layout

The map CSV and JSON drive a lot of derived parameters that you can't tune
independently — they're inferred from the cell layout.

| Map-derived value           | What it controls                          | KPIs affected                            |
|-----------------------------|-------------------------------------------|------------------------------------------|
| `num_goals`                 | Pickerwall slot count                     | Throughput ceiling; `bin_throughput`    |
| `num storage shelves`       | Storage capacity                          | Displacement frequency                  |
| `num replenishment shelves` | Bin spawn capacity                        | Bottleneck if too few                   |
| `num picker spawn cells`    | Effective picker count                    | `pick_throughput` ceiling               |
| `num idle zone cells`       | Idle parking capacity                     | `clash_rate` (highway clearance)        |
| `bins_per_shelf`            | Inferred from `column_height`             | Displacement frequency, `bin_throughput`|
| Grid dimensions             | Travel distances                          | `distance_per_bot`, `bin_throughput`    |

| Knob                  | Default | KPIs affected | Where             |
|-----------------------|---------|---------------|-------------------|
| `TARWARE_MAP_NAME`    | `medium` (env), `tiny` (compose) | All — completely changes the layout | `__init__.py:11` |
| `TARWARE_MAP_CSV_PATH`| auto from name | All | `__init__.py:14-17` |
| `TARWARE_MAP_JSON_PATH` | auto from name | All — defines highways, idle zones, packaging | `__init__.py:18-21` |

> **Practical guidance:** When comparing KPIs across maps, normalise by
> `n_agvs × num_goals` to factor out fleet/wall-size differences.

---

## 5. Bin / storage parameters

The volume model — how much each bin can hold — affects how often bins are
"depleted" and routed back to replenishment.

| Knob                            | Default  | KPIs affected                                | Direction                                                      | Where             |
|---------------------------------|----------|-----------------------------------------------|----------------------------------------------------------------|-------------------|
| `TARWARE_BIN_VOLUME_FT3`        | `2.68`   | `bin_throughput`, `distance_per_bot`          | ↑ → fewer depletions → ↓ replenishment trips → ↑ throughput   | `warehouse.py:316` |
| `TARWARE_BIN_USABLE_FRACTION`   | `0.85`   | Same                                          | ↑ → more usable space per bin → ↓ depletion frequency         | `warehouse.py:318` |

---

## 6. Heuristic constants (currently hard-coded)

These are baked into `heuristic.py` and `warehouse.py` — no env var. Changing
them requires editing source.

| Constant                              | Value | Where                  | KPIs affected                                  | Direction                                          |
|---------------------------------------|-------|------------------------|-------------------------------------------------|----------------------------------------------------|
| `DISPLACEMENT_THRESHOLD`              | `0.7` | `heuristic.py:228`     | `bin_throughput`, `distance_per_bot`            | ↑ → wait longer to displace → wall fills, lower throughput; ↓ → premature displacement, wasted moves |
| `_STUCK_THRESHOLD`                    | `5`   | `warehouse.py:33`      | `stucks`, indirectly `bin_throughput`           | ↑ → fewer stuck-resolutions, slower deadlock recovery; ↓ → more aggressive reroutes |
| `_FIXING_CLASH_TIME`                  | `4`   | `warehouse.py:32`      | `clash_rate`, `bin_throughput`                  | ↑ → longer cooldown after clash → fewer chained clashes but slower recovery |
| `_PICKER_BLOCKED_REROUTE_THRESHOLD`   | `4`   | `warehouse.py:34`      | `pick_throughput`                               | ↑ → pickers wait longer before detouring; ↓ → more picker movement / clashes |
| `_PICK_TICKS`                         | `3`   | `warehouse.py:35`      | `pick_throughput` (picker bottleneck)           | ↑ → pickers spend longer per claim → bottleneck   |

### Heuristic policy choices (no constant — structural)

- **AGV claim ordering:** `(-sku_backlog, -on_replenishment, item.id)` —
  see `heuristic.py:_request_queue_priority`. Changing the sort key changes
  which orders get serviced first.
- **Idle parking:** `MissionType.IDLE` is dispatched whenever an AGV is free
  *and* `idle_zone_action_ids` is non-empty. Maps without idle cells get
  no parking — AGVs sit on highways → can affect `clash_rate`.
- **Pickerwall vs storage delivery:** delivery always targets the
  closest-reachable pickerwall slot via `find_agv_path_to_goal_entry`;
  there is no override.

---

## 7. Picker model (only matters when `num_pickers > 0`)

These don't touch AGV-only KPIs (utilization, distance) but do change picker
throughput, which can become the bottleneck.

| Knob                                  | Default      | Effect                                            | Where                |
|---------------------------------------|--------------|---------------------------------------------------|----------------------|
| `TARWARE_PICKER_POLICY`               | `fifo`       | Order in which pickers claim deliveries           | `warehouse.py:334`   |
| `TARWARE_PICKER_ZONE_OVERFLOW`        | `adjacent`   | What pickers do when their zone is full           | `warehouse.py:335`   |
| `TARWARE_PICKER_STALL_PROBABILITY`    | `0.0`        | Random per-step stall chance                      | `warehouse.py:336`   |
| `TARWARE_PICKER_USE_SKU_SIZE_TIME`    | `0`          | Whether pick time scales with SKU volume          | `warehouse.py:337`   |
| `TARWARE_PICK_BASE_TICKS`             | `3`          | Base step count to pick one bin                   | `warehouse.py:342`   |
| `TARWARE_PICK_UNIT_CUBE_TICK_SCALE`   | `1.0`        | Multiplier when SKU-size time is on               | `warehouse.py:343`   |
| `TARWARE_PICK_BASE_SECONDS`           | (model-dep)  | Base seconds in physical-time model               | `human_factors.py:146` |
| `TARWARE_PICK_UNIT_CUBE_SECONDS_SCALE`| (model-dep)  | Volume scaling in physical-time model             | `human_factors.py:152` |
| `TARWARE_PICK_QUANTITY_EXTRA_SECONDS` | `1.0`        | Per-unit overhead                                 | `human_factors.py:157` |

### Human factors model

If pickers are enabled and human-factors fatigue is on, picker speeds
degrade over time according to an effort profile (Zhao et al. by default).

| Knob                                       | Default     | Effect                                | Where                 |
|--------------------------------------------|-------------|----------------------------------------|-----------------------|
| `TARWARE_HF_ENABLED`                       | `1`         | Toggle the entire fatigue model        | `human_factors.py:115`|
| `TARWARE_HF_MODEL`                         | `zhao`      | Which fatigue model to load            | `human_factors.py:116`|
| `TARWARE_HF_DEFAULT_PROFILE`               | `medium`    | Picker fitness profile                 | `human_factors.py:125`|
| `TARWARE_HF_FATIGUE_MIN` / `_MAX`          | `0` / `100` | Range of fatigue accumulation          | `human_factors.py:119-120` |
| `TARWARE_HF_DEFAULT_PROFILE_BY_MAP`        | `{}`        | Per-map override JSON                  | `human_factors.py:124`|
| `TARWARE_HF_PICKER_PROFILE_OVERRIDES`      | `{}`        | Per-picker override JSON               | `human_factors.py:129`|
| `TARWARE_HF_PROFILE_<NAME>`                | n/a         | Override individual profile params     | `human_factors.py:417`|

---

## 8. Reward type (`global_return` only)

| Knob                  | Default     | KPIs affected   | Where             |
|-----------------------|-------------|------------------|-------------------|
| `TARWARE_REWARD_TYPE` | `global` (env), `individual` (compose) | Only `global_return` | `__init__.py:58` |

`individual` and `global` change how rewards are aggregated across agents but
don't affect deliveries / utilisation / clashes. Only matters if you're
training an RL policy or comparing reward curves.

---

## 9. Episode count and seeds

| Knob              | Default | KPIs affected                                    | Direction                                                |
|-------------------|---------|---------------------------------------------------|----------------------------------------------------------|
| `num_episodes`    | `3` (Streamlit default)  | Statistical confidence; per-episode dashboard   | ↑ → more samples, smoother averages, longer wall-clock  |
| Seed strategy     | `seed=episode` (sequential 0,1,2…) | All — different seeds → different order patterns | Different seed → different scenario, not repeated trial |

> **Important:** server.py uses sequential seeds. Each episode is a *different*
> order pattern, not a repeat of the same scenario with different RNG. If you
> want repeated-trial averaging (same scenario, statistical noise only), edit
> `simulation_loop` to pass a fixed seed.

---

## 10. Knobs with no KPI effect

For completeness — these are present but don't move any number on the
dashboard.

| Knob                          | Effect                              |
|-------------------------------|--------------------------------------|
| `TARWARE_OBS_TYPE`            | Observation space (RL only); heuristic ignores |
| `TARWARE_RENDER_TILE_SIZE`    | Frame pixel size                     |
| `TARWARE_RENDER_WIDTH/HEIGHT` | Auto-fit canvas (frame size only)    |
| `FRAME_SCALE`                 | JPEG resize factor (memory only)     |
| `TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW` | Render overlay         |
| `TARWARE_RENDER_PHYSICAL_TIME_OVERLAY`     | Render overlay         |

---

## Recommended workflow for KPI experiments

1. **Pin everything else.** Fix `TARWARE_MAP_NAME`, `TARWARE_ORDER_CSV_PATH`,
   `TARWARE_STEPS_PER_SIMULATED_SECOND`, and `num_episodes ≥ 5` before
   sweeping a single knob.
2. **Sweep one variable at a time** through 3-5 values; record per-episode
   stats from `/episode_stats`.
3. **Compare on the right scale.** AGV-side throughput knobs (AGVs, queue
   size): use `bin_throughput` and `deliveries`. Picker-side throughput
   knobs (pickers, pick-time constants): use `pick_throughput` and
   `items_picked`. Routing knobs (map, ABC, bin size): use
   `distance_per_bot`. Contention knobs (AGV count, stuck threshold): use
   `clash_rate` and `stucks`.
4. **Acknowledge the seed dependency.** With sequential seeds, average over
   ≥ 5 episodes to get stable estimates. With fixed seed, you measure the
   policy's deterministic behaviour on one scenario only.

---

## Files

- `simulation/tarware/tarware/__init__.py` — env-registration knobs
- `simulation/tarware/tarware/warehouse.py` — env constants, hard-coded thresholds, picker tunables
- `simulation/tarware/tarware/heuristic.py` — `DISPLACEMENT_THRESHOLD`, dispatch policy
- `simulation/tarware/tarware/human_factors.py` — physical-speed and fatigue model
- `simulation/tarware/tarware/order_sequencer.py` — order release timing
- `simulation/server.py::simulation_loop` — episode count and seed strategy
- `compose.yaml` — runtime defaults forwarded to spawned sim containers
