#!/usr/bin/env bash
# start.sh — start sem-service (Docker)
set -e
cd "$(dirname "$0")"

echo "[start.sh] Starting sem-service..."
cd sem-service && docker compose up -d && cd ..
echo ""
echo "  SEM Service → http://localhost:3000"

