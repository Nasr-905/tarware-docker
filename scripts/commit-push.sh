#!/usr/bin/env bash
#
# Stage all changes in the parent repo, commit with the given message under
# the Tarware Capstone identity, and push to the current branch on origin.
# Never force-pushes.
#
# Usage:
#   ./scripts/commit-push.sh "<commit message>"
#
# Notes:
#   - Stages everything via `git add -A`. .env / *.pyc / __pycache__ etc.
#     are gitignored already; review the staged set before re-running if
#     you care about granularity.
#   - Submodule changes (under simulation/tarware/) are NOT committed by
#     this script; only the parent's pointer to the submodule HEAD is
#     staged. Push the submodule's own commits with:
#         git -C simulation/tarware push origin <branch>
#   - Identity is set per-invocation (no `git config` writes). No
#     Co-Authored-By trailer is added.

set -euo pipefail

if [[ $# -lt 1 || -z "${1// /}" ]]; then
    echo "usage: $0 \"<commit message>\"" >&2
    exit 1
fi

MESSAGE="$1"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git -c user.name="Tarware Capstone" \
        -c user.email="capstone@tarware.local" \
        commit -m "$MESSAGE"
else
    echo "[commit-push] working tree clean; pushing existing commits"
fi

# Heads up if the submodule has unpushed work — the parent's pointer is
# only useful once the SHA it references is reachable on the submodule's
# remote.
if git -C simulation/tarware rev-parse --abbrev-ref @{u} >/dev/null 2>&1; then
    SUB_AHEAD="$(git -C simulation/tarware rev-list --count @{u}..HEAD 2>/dev/null || echo 0)"
    if [[ "$SUB_AHEAD" -gt 0 ]]; then
        echo "[commit-push] WARNING: simulation/tarware is $SUB_AHEAD commit(s) ahead of its upstream."
        echo "[commit-push]          Push the submodule first or the parent's pointer will be unresolvable for collaborators."
    fi
fi

git push origin "$BRANCH"
