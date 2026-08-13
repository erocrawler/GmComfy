# Configuration

This document outlines the environment variables available for configuring the `worker-comfyui`.

## General Configuration

| Environment Variable | Description                                                                                                                                                                                                                  | Default |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `REFRESH_WORKER`     | When `true`, the worker pod will stop after each completed job to ensure a clean state for the next job. See the [RunPod documentation](https://docs.runpod.io/docs/handler-additional-controls#refresh-worker) for details. | `false` |
| `SERVE_API_LOCALLY`  | When `true`, enables a local HTTP server simulating the RunPod environment for development and testing. See the [Development Guide](development.md#local-api) for more details.                                              | `false` |
| `COMFY_ORG_API_KEY`  | Comfy.org API key to enable ComfyUI API Nodes. If set, it is sent with each workflow; clients can override per request via `input.api_key_comfy_org`.                                                                        | –       |

## Memory Management Configuration

These settings control automatic memory reclamation for ComfyUI between jobs. The check runs once per job (pre-job) and only while ComfyUI's queue is idle, so in-flight generations are never disturbed.

| Environment Variable              | Description                                                                                                                                                                                                                                                                                                                                                         | Default  |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `COMFY_MEMORY_MONITOR`            | Enable/disable the memory monitor entirely. When enabled, the worker checks system memory before each job and calls ComfyUI's `/free` endpoint (unload models / free memory) when usage exceeds the limit.                                                                                                                                                           | `true`   |
| `COMFY_MEMORY_LIMIT_PERCENT`      | System memory usage (in percent) that triggers a `/free` call. Uses cgroup limits when available (Docker-aware), falling back to host memory info.                                                                                                                                                                                                                  | `85`     |
| `COMFY_MEMORY_FREE_COOLDOWN_S`    | Minimum seconds between `/free` calls, preventing hammering the endpoint.                                                                                                                                                                                                                                                                                           | `300`    |
| `COMFY_FREE_TIMEOUT_S`            | Timeout in seconds for the `/free` and `/queue` HTTP calls.                                                                                                                                                                                                                                                                                                        | `10`     |
| `COMFY_MEMORY_RESTART`            | When `true`, escalate to a full ComfyUI restart if memory is still above the limit a short while after `/free` (the leak is likely inside ComfyUI itself). The process is killed and the `start.sh` supervisor loop restarts it with a clean process.                                                                                                                | `false`  |
| `COMFY_MEMORY_VERIFY_DELAY_S`     | Seconds to wait after the `/free` call before re-checking memory to decide whether a restart is needed. Only used when `COMFY_MEMORY_RESTART=true`.                                                                                                                                                                                                                 | `10`     |
| `COMFY_RESTART_MIN_UPTIME_S`      | Minimum seconds that must elapse between ComfyUI restarts (anti restart-loop guard, e.g. page-cache pressure right after a restart). Only used when `COMFY_MEMORY_RESTART=true`.                                                                                                                                                                                    | `300`    |
| `COMFY_RESTART_WAIT_MAX_RETRIES`  | How many times to poll the ComfyUI HTTP API (at `COMFY_API_AVAILABLE_INTERVAL_MS` intervals) while waiting for it to come back up after a restart before giving up on the current job.                                                                                                                                                                               | `120`    |
| `COMFY_RESTART_DELAY_S`           | Delay in seconds the `start.sh` supervisor loop waits before restarting ComfyUI after the process exits.                                                                                                                                                                                                                                                            | `2`      |
| `COMFY_API_AVAILABLE_INTERVAL_MS` | Time in milliseconds between checks while waiting for the ComfyUI API to become available (also used by the restart wait).                                                                                                                                                                                                                                          | `500`    |
| `COMFY_API_AVAILABLE_MAX_RETRIES` | Maximum number of API availability checks before the handler gives up.                                                                                                                                                                                                                                                                                              | `500`    |

> [!NOTE]
> **Restart escalation (`COMFY_MEMORY_RESTART=true`):**
> - The worker first tries the cheap path: POST `/free` (unload models / free memory).
> - After `COMFY_MEMORY_VERIFY_DELAY_S` it re-checks memory. If usage is still above `COMFY_MEMORY_LIMIT_PERCENT`, the leak is likely inside ComfyUI itself (cached buffers that `/free` cannot release).
> - It then kills the ComfyUI process; `start.sh` runs ComfyUI inside a supervisor loop and restarts it automatically.
> - The worker waits for ComfyUI to become reachable again (up to `COMFY_RESTART_WAIT_MAX_RETRIES` polls) before continuing with the job, so the job is delayed by the restart rather than lost.
> - `COMFY_RESTART_MIN_UPTIME_S` prevents restart loops.

## Logging Configuration

| Environment Variable | Description                                                                                                                                                      | Default |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `COMFY_LOG_LEVEL`    | Controls ComfyUI's internal logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Use `DEBUG` for troubleshooting, `INFO` for production. | `DEBUG` |

## Debugging Configuration

| Environment Variable           | Description                                                                                                            | Default |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| `WEBSOCKET_RECONNECT_ATTEMPTS` | Number of websocket reconnection attempts when connection drops during job execution.                                  | `5`     |
| `WEBSOCKET_RECONNECT_DELAY_S`  | Delay in seconds between websocket reconnection attempts.                                                              | `3`     |
| `WEBSOCKET_TRACE`              | Enable low-level websocket frame tracing for protocol debugging. Set to `true` only when diagnosing connection issues. | `false` |

> [!TIP] > **For troubleshooting:** Set `COMFY_LOG_LEVEL=DEBUG` to get detailed logs when ComfyUI crashes or behaves unexpectedly. This helps identify the exact point of failure in your workflows.

## Webhook Configuration

These settings control the behavior of webhook notifications sent when using the `callback_url` parameter in job requests. Webhooks are used to notify external systems when jobs complete, fail, or report progress.

| Environment Variable           | Description                                                                                                                                                           | Default |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `I2V_WEBHOOK_RETRIES`          | Number of retry attempts for final webhook delivery (completion/failure notifications). Does not apply to progress updates which are fire-and-forget.                 | `5`     |
| `I2V_WEBHOOK_BACKOFF_S`        | Base delay in seconds for exponential backoff between webhook retry attempts. Actual delay increases exponentially: base × 2^(attempt-1) + random jitter.            | `2.0`   |
| `I2V_WEBHOOK_MAX_BACKOFF_S`    | Maximum delay in seconds between webhook retry attempts, caps the exponential backoff to prevent excessively long waits.                                              | `60.0`  |
| `I2V_WEBHOOK_TIMEOUT_S`        | Timeout in seconds for webhook HTTP requests. Increase this if your webhook endpoint needs more time to respond.                                                     | `30`    |

> [!NOTE]
> **Webhook Retry Behavior:**
> - **Final notifications** (completion/failure) use exponential backoff with jitter and retry on connection errors, timeouts, and other network issues.
> - **Final notifications run in a background thread**, allowing the handler to return immediately and process the next job on the GPU without waiting for webhook retries to complete.
> - **Progress notifications** are sent once without retries (fire-and-forget) to avoid delays in job processing.
> - **Error notifications** during job processing use synchronous retries to ensure delivery before returning error status.
> - The retry mechanism handles common network errors including `ConnectionError`, `Timeout`, and general `RequestException` cases.
> - Example backoff sequence with defaults (2s base): 2s → 4s → 8s → 16s → 32s (capped at 60s if `I2V_WEBHOOK_MAX_BACKOFF_S` is set)

## AWS S3 Upload Configuration

Configure these variables **only** if you want the worker to upload generated images directly to an AWS S3 bucket. If these are not set, images will be returned as base64-encoded strings in the API response.

- **Prerequisites:**
  - An AWS S3 bucket in your desired region.
  - An AWS IAM user with programmatic access (Access Key ID and Secret Access Key).
  - Permissions attached to the IAM user allowing `s3:PutObject` (and potentially `s3:PutObjectAcl` if you need specific ACLs) on the target bucket.

| Environment Variable       | Description                                                                                                                             | Example                                                    |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `BUCKET_ENDPOINT_URL`      | The full endpoint URL of your S3 bucket. **Must be set to enable S3 upload.**                                                           | `https://<your-bucket-name>.s3.<aws-region>.amazonaws.com` |
| `BUCKET_ACCESS_KEY_ID`     | Your AWS access key ID associated with the IAM user that has write permissions to the bucket. Required if `BUCKET_ENDPOINT_URL` is set. | `AKIAIOSFODNN7EXAMPLE`                                     |
| `BUCKET_SECRET_ACCESS_KEY` | Your AWS secret access key associated with the IAM user. Required if `BUCKET_ENDPOINT_URL` is set.                                      | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`                 |

**Note:** Upload uses the `runpod` Python library helper `rp_upload.upload_image`, which handles creating a unique path within the bucket based on the `job_id`.

### Example S3 Response

If the S3 environment variables (`BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY`) are correctly configured, a successful job response will look similar to this:

```json
{
  "id": "sync-uuid-string",
  "status": "COMPLETED",
  "output": {
    "images": [
      {
        "filename": "ComfyUI_00001_.png",
        "type": "s3_url",
        "data": "https://your-bucket-name.s3.your-region.amazonaws.com/sync-uuid-string/ComfyUI_00001_.png"
      }
      // Additional images generated by the workflow would appear here
    ]
    // The "errors" key might be present here if non-fatal issues occurred
  },
  "delayTime": 123,
  "executionTime": 4567
}
```

The `data` field contains the presigned URL to the uploaded image file in your S3 bucket. The path usually includes the job ID.
