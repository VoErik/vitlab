#!/usr/bin/env bash
# Launch the vitlab Explorer: FastAPI backend (:8000) + Vite frontend (:5173).
# Precompute at least one atlas first:
#   uv run python scripts/app_precompute_atlas.py --model ... --bank ... --dataset ... --site ...
set -euo pipefail
cd "$(dirname "$0")"

echo "==> starting backend on http://localhost:8000"
uv run uvicorn app.backend.main:app --reload --port 8000 &
BACKEND=$!
trap "kill $BACKEND 2>/dev/null || true" EXIT

if [ -d app/frontend/node_modules ]; then
  echo "==> starting frontend on http://localhost:5173"
  (cd app/frontend && npm run dev)
else
  echo "!! frontend deps not installed. Run: (cd app/frontend && npm install)"
  echo "   backend is up; API docs at http://localhost:8000/docs"
  wait $BACKEND
fi
