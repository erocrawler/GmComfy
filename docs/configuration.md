# Configuration

This document outlines the environment variables available for configuring the `worker-comfyui`.

## General Configuration

| Environment Variable | Description                                                                                                                                                                                                                  | Default |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `REFRESH_WORKER`     | When `true`, the worker pod will stop after each completed job to ensure a clean state for the next job. See the [RunPod documentation](https://docs.runpod.io/docs/handler-additional-controls#refresh-worker) for details. | `false` |
| `SERVE_API_LOCALLY`  | When `true`, enables a local HTTP server simulating the RunPod environment for development and testing. See the [Development Guide](development.md#local-api) for more details.                                              | `false` |
| `COMFY_ORG_API_KEY`  | Comfy.org API key to enable ComfyUI API Nodes. If set, it is sent with each workflow; clients can override per request via `input.api_key_comfy_org`.                                                                        | –       |

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
