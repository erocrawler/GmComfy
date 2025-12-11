#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is INFO.
: "${COMFY_LOG_LEVEL:=INFO}"

# Support enabling a high-VRAM mode via env var `COMFY_HIGH_VRAM`
# Usage: set COMFY_HIGH_VRAM=true in environment to enable.
COMFY_HIGH_VRAM=${COMFY_HIGH_VRAM:-false}

if [ "$COMFY_HIGH_VRAM" = "true" ]; then
    echo "worker-comfyui: High VRAM mode enabled"
    HIGH_VRAM_ARG="--highvram"
else
    HIGH_VRAM_ARG=""
fi

# Support enabling/disabling sage attention via env var `COMFY_USE_SAGE_ATTENTION`
# Usage: set COMFY_USE_SAGE_ATTENTION=false in environment to disable (default is true).
COMFY_USE_SAGE_ATTENTION=${COMFY_USE_SAGE_ATTENTION:-true}

if [ "$COMFY_USE_SAGE_ATTENTION" = "true" ]; then
    echo "worker-comfyui: Sage attention enabled"
    SAGE_ATTENTION_ARG="--use-sage-attention"
else
    SAGE_ATTENTION_ARG=""
fi

# Support enabling flash attention via env var `USE_FLASH_ATTN`
# Usage: set USE_FLASH_ATTN=true in environment to enable.
USE_FLASH_ATTN=${USE_FLASH_ATTN:-false}

if [ "$USE_FLASH_ATTN" = "true" ]; then
    echo "worker-comfyui: Flash attention enabled"
    FLASH_ATTN_ARG="--use-flash-attention"
else
    FLASH_ATTN_ARG=""
fi

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata ${HIGH_VRAM_ARG} ${SAGE_ATTENTION_ARG} ${FLASH_ATTN_ARG} --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata ${HIGH_VRAM_ARG} ${SAGE_ATTENTION_ARG} ${FLASH_ATTN_ARG} --verbose "${COMFY_LOG_LEVEL}" --log-stdout &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi