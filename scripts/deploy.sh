#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Diabetes Tracker — Docker Deploy ==="

log "[1/3] Building images..."
docker compose -f docker/docker-compose.yml build

log "[2/3] Starting services..."
docker compose -f docker/docker-compose.yml up -d

log "[3/3] Waiting for health checks..."
sleep 10
docker compose -f docker/docker-compose.yml ps

log "API     -> http://localhost:5001/health"
log "MLflow  -> http://localhost:5000"
