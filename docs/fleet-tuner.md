# Fleet Sizing Tuner

A second Streamlit page that sweeps `(num_agvs, num_pickers)` combinations
for a chosen map and reports the **Pareto-optimal** trade-off between
throughput (chosen via the **Optimization metric** dropdown — either
`bin_throughput` or `pick_throughput`) and total fleet size. Uses headless
sim runs (no frame capture) so each combo is dramatically faster than a
normal playback session.

> **Verify before trusting.** Performance numbers and access-control
> details depend on environment specifics (Pangolin proxy header behaviour,
> Streamlit version, host CPU). Re-test for your deployment.

---

## When to use it

- Deciding how many AGVs / pickers to provision for a given warehouse layout.
- Sanity-checking that adding agents past a certain point stops helping.
- Producing a defensible fleet-size recommendation backed by simulation data.

## What "Pareto-optimal" means here

A combo $(a, p)$ is Pareto-optimal if no other completed combo has at the
same time a *higher or equal* value on the chosen metric AND a *lower or
equal* `total_agents = num_agvs + num_pickers`. The Pareto frontier is the
set of combos where you can't improve throughput without adding more
agents (and vice versa). The tuner highlights these in red on the scatter
plot.

The frontier is metric-specific. With `bin_throughput`, the optimum
typically lands at zero pickers (pickers don't deliver bins, so they are
strictly dominated). With `pick_throughput`, the frontier includes combos
that balance AGVs and pickers — adding an AGV without a picker to consume
its deliveries doesn't help.

There is no single "winner" — picking from the frontier requires a domain
trade-off (capital cost vs throughput target) that the tool deliberately
leaves to the operator.

---

## Access control

The tuner is resource-intensive (each combo spawns a sim container) and
should not be available to public users reaching Streamlit through Pangolin.
Two gates protect it:

### Gate 1 — env var (the actual lock)

`TUNER_ENABLED=1` must be set in the Streamlit container's environment.
Default in `compose.yaml`. Remove or set to `0` for any public deployment.

### Gate 2 — private-IP check (best-effort)

`streamlit/access.py::tuner_access_state` reads `X-Forwarded-For` from the
request via `st.context.headers` and verifies the first IP is in a private
range (RFC 1918 / loopback / link-local). Direct host access (no XFF
header) is treated as local.

### Combined behaviour

| State                                   | Page visible? | Controls usable? |
|-----------------------------------------|---------------|-------------------|
| `TUNER_ENABLED` not set                 | Yes           | No (greyed out)   |
| `TUNER_ENABLED=1` + public IP           | Yes           | No (greyed out)   |
| `TUNER_ENABLED=1` + private/local IP    | Yes           | Yes               |

The page is always shown in the sidebar so users know the feature exists.
A warning banner explains *why* the controls are disabled when access is
denied.

### Hardening for production

If you care about defeat-resistance:
- Verify Pangolin's `X-Forwarded-For` cannot be set by the upstream client
  (test by manually forging the header).
- Or run a separate Streamlit instance on an internal-only port for the
  tuner, behind compose `profiles:` so it isn't started in production.

---

## Tuner controls

| Control                  | Default | Effect                                                        |
|--------------------------|---------|---------------------------------------------------------------|
| Map                      | first available | Map CSV/JSON used by every combo                       |
| Optimization metric      | `bin_throughput` | Drives plateau detection AND Pareto frontier (see below) |
| AGVs (min, max)          | 1, 20   | Range to sweep (UI cap 200)                                  |
| Pickers (min, max)       | 0, 10   | Range to sweep (UI cap 200)                                  |
| Episodes per combo       | 3       | Seeds per combo; KPIs are averaged across them                 |
| Concurrent sims          | 12      | Max simultaneous sim containers; each uses ~1 CPU core         |
| Plateau tolerance (%)    | 2.0     | Metric gain below this % of best-so-far in the row counts as a plateau hit |
| Plateau streak           | 2       | Consecutive plateau hits before stopping that picker row      |

### Optimization metric

| Metric            | What it measures                              | Pick when …                                       |
|-------------------|-----------------------------------------------|---------------------------------------------------|
| `bin_throughput`  | AGV-side: bins delivered to pickerwall per simulated hour | Sizing the AGV fleet only; pickers are out of scope or not modelled |
| `pick_throughput` | Picker-side: units physically picked per simulated hour    | Sizing the joint AGV + picker fleet; you care about end-to-end output |

The dropdown changes:
- **Plateau detection** — the row's plateau counter ticks on the chosen metric.
- **Pareto frontier** — the dominance check uses the chosen metric on the y-axis.
- **Heatmap colour** and **table sort** — both follow the chosen metric.

Both metrics are still computed and displayed in the table regardless of
which one is selected — only what *drives* pruning and Pareto changes.

If you select `pick_throughput` with **Pickers (min) = 0**, the tuner
refuses to start: with no pickers, `items_picked` is always zero and the
frontier degenerates to a single point. Either raise Pickers (min) ≥ 1 or
switch the metric to `bin_throughput`.

### How pruning works

For each fixed picker count, AGVs are swept in ascending order. After each
combo (using the chosen optimization metric `m`):
1. If `result[m] <= prev_best * (1 + tolerance)`, increment plateau counter.
2. If counter reaches `plateau_streak`, drop all remaining AGV combos for
   *this* picker count from the queue.
3. If `result[m]` improves, reset counter and update `prev_best`.

Picker rows are not pruned — they run to completion of the (possibly
truncated) AGV row. Effective grid coverage is typically 30-60% of the raw
sweep depending on where saturation occurs.

### Concurrency model

Row-parallel, sequential-within-row:

- Each picker count is a "row" with its own ascending AGV queue.
- Multiple rows run **concurrently** up to `Concurrent sims`.
- Within a single row, AGV combos run **sequentially** so plateau detection
  observes results in order (no wasted runs from late arrivals).

On each tuner tick (~2 s):

1. Poll every in-flight session's `/status`.
2. For each finished session: pull `/episode_stats`, compute averages,
   apply plateau check to that row, advance the row's queue.
3. Spawn new sessions for any row that has work and no in-flight session,
   up to the concurrency cap.

This means up to `min(num_picker_rows, concurrency)` sims run at once.
With `picker_min=0, picker_max=10, concurrency=12`, you'll see up to 11
rows in flight (one per picker count).

### Hardware sizing for the concurrency cap

Each headless sim container uses approximately 1 CPU core (Python heuristic
+ pyastar2d C extension; rendering is skipped). RAM per container is
~200-400 MB. Recommended cap:

| Host                   | Recommended cap |
|------------------------|-----------------|
| 4-core / 16 GB         | 3               |
| 8-core / 32 GB         | 6-7             |
| 16-core / 64 GB (5950X)| 12              |
| 32-core / 128 GB       | 24              |

Leave 2 cores for the manager + streamlit + Docker daemon.

---

## Performance expectations

Headless mode skips `env.render()`, JPEG encoding, and the `all_frames`
buffer entirely — KPI computation is the only per-step work besides A* and
env step. Rough wall-clock per combo on a 4-core host:

| Map        | Grid size | AGVs | Episodes | Wall time per combo |
|------------|-----------|------|----------|----------------------|
| `tiny_dhl` | ~18×18    | 1-8  | 3        | 5-15 s               |
| `small_dhl`| ~10×34    | 1-8  | 3        | 10-30 s              |
| `medium_dhl`| ~22×45   | 1-12 | 3        | 30-90 s              |
| `large_dhl`| ~38×61    | 1-19 | 3        | 90-300 s             |

With concurrency=1 (sequential), a full 8×5 sweep on `tiny_dhl` with
pruning typically completes in 3-5 minutes; the same sweep on `large_dhl`
can take 30+ minutes. With concurrency=12 on a 5950X-class host, a
20×10 sweep on `large_dhl` completes in roughly 30-45 minutes.

---

## Outputs

- **Heatmap** — rows = pickers, cols = AGVs, colour = mean of the chosen
  metric (`bin_throughput` or `pick_throughput`).
- **Pareto scatter** — chosen metric vs `total_agents`. Red points are
  Pareto-optimal; grey points are dominated.
- **Sortable table** — every completed combo with both `bin_throughput`
  and `pick_throughput`, `agv_utilization`, `clash_rate`,
  `distance_per_bot`, `deliveries`, `items_picked`, `stucks`, episodes
  completed, and the `pareto_optimal` flag. Sorted by the chosen metric.
- **CSV download** — same data as the table, suitable for further analysis.
  Filename includes the chosen metric so different runs don't overwrite.

---

## Troubleshooting

| Symptom                                              | Likely cause                                                  | Fix                                                     |
|------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------|
| Sidebar page exists but everything is greyed out     | One of the two access gates is failing                        | Banner at top of page tells you which                   |
| All combos report `bin_throughput: 0`                | Sim crashed at episode start (e.g. picker count > spawn cells, but env should clamp) | Inspect `docker logs tarware_sim_<id>` mid-run         |
| Pareto frontier is *always* at picker = 0            | Optimization metric is `bin_throughput` — pickers don't deliver bins, so they're strictly dominated on this axis | Switch the metric to `pick_throughput` (and set Pickers (min) ≥ 1) |
| All combos report `pick_throughput: 0`               | Pickers (min) = 0 *or* the map has no picker spawn cells     | Raise Pickers (min) ≥ 1; verify the map JSON has picker spawn cells |
| "no_episode_stats" error column                      | Sim finished but `/episode_stats` returned an empty list      | Sim likely crashed mid-run; check sim logs              |
| Plateau detection stops the sweep too early          | Tolerance too tight, or noise floor exceeds the gain          | Increase tolerance from 2% → 5%; raise episodes/combo   |
| Sweep runs forever                                   | Plateau streak too high, or genuinely no plateau              | Reduce streak from 2 → 1, or shrink AGVs (max)          |
| Stop button doesn't stop immediately                 | Mid-combo, the in-flight session is force-deleted on stop, but the polling loop still has to return | Wait one poll interval (~2 s)                           |

---

## Implementation notes

- `simulation/server.py` reads `TARWARE_HEADLESS=1` once at module load and
  gates frame rendering throughout. Stats accumulation runs unconditionally,
  so headless KPIs are identical to non-headless KPIs (same env, same
  heuristic, same seed).
- `manager/manager.py` accepts `headless: bool` on `POST /session`; sets
  `TARWARE_HEADLESS=1` per-session in the spawned container's environment.
- `streamlit/pages/2_Fleet_Tuner.py` runs one combo per Streamlit rerun.
  Session state holds the pending queue, completed results, and per-row
  plateau counters across reruns. Stop force-deletes the in-flight session
  via `DELETE /session/{id}`.
- The plan / approach details: see `/home/coder/.claude/plans/jazzy-leaping-snail.md`.

## Files

- `streamlit/pages/2_Fleet_Tuner.py` — the tuner page (controls, run loop, charts, export)
- `streamlit/access.py` — `tuner_access_state()` helper
- `simulation/server.py` — headless rendering gate
- `manager/manager.py` — `headless` flag on `SessionRequest`
- `compose.yaml` — `TUNER_ENABLED` default
