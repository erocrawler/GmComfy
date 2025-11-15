#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# Support enabling a high-VRAM mode via env var `COMFY_HIGH_VRAM`
# Usage: set COMFY_HIGH_VRAM=true in environment to enable.
COMFY_HIGH_VRAM=${COMFY_HIGH_VRAM:-false}

if [ "$COMFY_HIGH_VRAM" = "true" ]; then
    echo "worker-comfyui: High VRAM mode enabled"
    HIGH_VRAM_ARG="--highvram"
else
    HIGH_VRAM_ARG=""
fi

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata ${HIGH_VRAM_ARG} --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata ${HIGH_VRAM_ARG} --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi