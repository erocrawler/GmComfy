#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is INFO.
: "${COMFY_LOG_LEVEL:=INFO}"

# Echo the given CLI flag if the env var (with default) resolves to "true".
# Usage: add_flag <env_var> <default> <cli_flag>
add_flag() {
    local env_var="$1" default="$2" flag="$3"
    local value="${!env_var:-$default}"
    if [ "$value" = "true" ]; then
        printf '%s' "$flag"
    fi
}

# Collect the ComfyUI CLI arguments common to every launch path.
COMFY_ARGS="--disable-auto-launch --disable-metadata --fast-disk --disable-pinned-memory"

# Support enabling a high-VRAM mode via env var `COMFY_HIGH_VRAM`
# Usage: set COMFY_HIGH_VRAM=true in environment to enable.
COMFY_ARGS="${COMFY_ARGS} $(add_flag COMFY_HIGH_VRAM false --highvram)"

# Support enabling/disabling sage attention via env var `COMFY_USE_SAGE_ATTENTION`
# Usage: set COMFY_USE_SAGE_ATTENTION=false in environment to disable (default is true).
COMFY_ARGS="${COMFY_ARGS} $(add_flag COMFY_USE_SAGE_ATTENTION true --use-sage-attention)"

# Support enabling flash attention via env var `USE_FLASH_ATTN`
# Usage: set USE_FLASH_ATTN=true in environment to enable.
COMFY_ARGS="${COMFY_ARGS} $(add_flag USE_FLASH_ATTN false --use-flash-attention)"

COMFY_ARGS="${COMFY_ARGS} --verbose ${COMFY_LOG_LEVEL} --log-stdout"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    COMFY_ARGS="${COMFY_ARGS} --listen"
    HANDLER_ARGS="--rp_serve_api --rp_api_host=0.0.0.0"
else
    HANDLER_ARGS=""
fi

# Restart delay for the ComfyUI supervisor loop (seconds).
: "${COMFY_RESTART_DELAY_S:=2}"

# Run ComfyUI inside a supervisor loop. In serverless mode there is no external
# process manager, so this loop plays that role: whenever the ComfyUI process
# exits - a crash, an OOM-kill, or an intentional kill from the handler's memory
# monitor (COMFY_MEMORY_RESTART=true) - it is restarted with a fresh process.
# Handlers poll the HTTP API (check_server) before each job, so they wait for
# the restart to complete.
restart_comfyui() {
    while true; do
        echo "worker-comfyui: Starting ComfyUI"
        python -u /comfyui/main.py ${COMFY_ARGS}
        exit_code=$?
        echo "worker-comfyui: ComfyUI exited (code ${exit_code}); restarting in ${COMFY_RESTART_DELAY_S}s..."
        sleep "${COMFY_RESTART_DELAY_S}"
    done
}

restart_comfyui &

echo "worker-comfyui: Starting RunPod Handler"
python -u /handler.py ${HANDLER_ARGS}