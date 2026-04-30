# Run Locally

## Prerequisites

- Docker + Docker Compose
- ~4 GB free RAM (more if you run the fleet tuner)

## Build images

The three images aren't pulled — you build them once locally:

```bash
docker build -t tarware-simulation:latest ./simulation
docker build -t tarware-manager:latest    ./manager
docker build -t tarware-streamlit:latest  ./streamlit
```

Rebuild only the one you changed; sim image is needed before any session can spawn.

## Start

```bash
docker compose up -d
```

Open the UI: <http://localhost:8501>

The `simulation` service uses the `build-only` profile and never starts itself — the manager spawns one `tarware_sim_*` container per browser tab on demand.

## Logs

```bash
docker compose logs -f manager streamlit
docker logs -f tarware_sim_<id>     # one specific sim
```

## Teardown

```bash
docker ps -a --filter name=tarware_sim --format '{{.ID}}' | xargs -r docker rm -f
docker compose down
```

The first command is required — sim containers aren't part of `compose.yaml`, so `compose down` alone leaves orphans.

## Reset between code changes

After editing `manager/`, `streamlit/`, or `simulation/`:

```bash
# rebuild the affected image, then:
docker compose up -d --no-deps --force-recreate <service>
```

If you edited `simulation/`, also kill any in-flight sim containers (`docker rm -f`) so new sessions pick up the new image.
