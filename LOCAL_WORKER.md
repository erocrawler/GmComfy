# GmAnimato Local Worker

A Python worker that polls the GmAnimato API for pending local jobs and processes them using ComfyUI.

## Overview

This worker is designed to work alongside the GmAnimato application's local job queue system. Instead of sending all jobs to RunPod serverless, jobs are routed to a local queue when capacity is available, and this worker processes them.

## Features

- **Automatic job polling** - Continuously polls `/api/worker/task` for pending jobs
- **Atomic job claiming** - Uses database-level locking to prevent race conditions with multiple workers
- **ComfyUI integration** - Reuses the existing `handler.py` logic for workflow execution
- **Webhook notifications** - Sends completion status back to GmAnimato via `/api/i2v-webhook/[id]`
- **Error handling** - Automatic retries with exponential backoff
- **Configurable** - Environment variables and CLI arguments for customization

## Prerequisites

- Python 3.8+
- ComfyUI running locally (default: `127.0.0.1:8188`)
- GmAnimato API accessible
- All dependencies from `requirements.txt` installed

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make the worker executable (Linux/Mac)
chmod +x local_worker.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GMANIMATO_API_URL` | `http://localhost:5173` | GmAnimato API base URL |
| `WORKER_POLL_INTERVAL` | `5` | Seconds between polls when no tasks available |
| `WORKER_MAX_RETRIES` | `3` | Maximum callback retry attempts |
| `COMFY_HOST` | `127.0.0.1:8188` | ComfyUI server address |
| `BUCKET_ENDPOINT_URL` | - | S3-compatible bucket endpoint (optional) |
| `BUCKET_NAME` | - | S3 bucket name (optional) |
| `BUCKET_ACCESS_KEY_ID` | - | S3 access key (optional) |
| `BUCKET_SECRET_ACCESS_KEY` | - | S3 secret key (optional) |

### Command Line Arguments

```bash
python local_worker.py --help
```

Options:
- `--api-url URL` - GmAnimato API URL
- `--poll-interval SECONDS` - Poll interval in seconds
- `--max-retries COUNT` - Maximum callback retry attempts
- `--debug` - Enable debug logging

## Usage

### Basic Usage

```bash
# Start the worker with default settings
python local_worker.py

# Start with custom API URL
python local_worker.py --api-url http://gmanimato.example.com

# Start with debug logging
python local_worker.py --debug
```

### Production Deployment

```bash
# Using environment variables
export GMANIMATO_API_URL=http://your-api-url
export WORKER_POLL_INTERVAL=3
export BUCKET_ENDPOINT_URL=https://your-s3-endpoint
export BUCKET_NAME=your-bucket
python local_worker.py
```

### Running Multiple Workers

You can run multiple worker instances to process jobs in parallel. The atomic job claiming mechanism ensures no job is processed twice:

```bash
# Terminal 1
python local_worker.py

# Terminal 2
python local_worker.py

# Terminal 3
python local_worker.py
```

Each worker will claim and process different jobs from the queue.

## How It Works

1. **Poll for Tasks**: Worker calls `GET /api/worker/task`
   - API atomically claims oldest pending local job
   - Job status is set to "processing"
   - Job details returned to worker

2. **Process Job**: Worker executes ComfyUI workflow
   - Downloads input image
   - Runs workflow through ComfyUI
   - Generates output video/images
   - Uploads to S3 or encodes as base64

3. **Send Completion**: Worker POSTs to `/api/i2v-webhook/[id]`
   - Sends job status (COMPLETED/FAILED)
   - Includes output files array
   - Includes error details if failed

4. **Repeat**: Worker polls for next task

## Workflow Data Format

The task returned by the API includes:

```json
{
  "id": "video-id",
  "job_id": "local-video-id",
  "original_image_url": "https://example.com/image.png",
  "prompt": "A beautiful video",
  "callback_url": "http://api/api/i2v-webhook/video-id",
  "workflow": { /* ComfyUI workflow JSON */ }
}
```

The worker transforms this into ComfyUI handler format:

```json
{
  "id": "video-id",
  "input": {
    "workflow": { /* workflow */ },
    "images": [
      {
        "name": "video-id.png",
        "image": "https://example.com/image.png"
      }
    ],
    "callback_url": "http://api/api/i2v-webhook/video-id"
  }
}
```

## Logging

The worker logs all activities:

```
2025-12-13 10:30:00 - local-worker - INFO - Starting local worker...
2025-12-13 10:30:05 - local-worker - INFO - Received task: clxxxx123
2025-12-13 10:30:05 - local-worker - INFO - Processing task clxxxx123
2025-12-13 10:30:05 - local-worker - INFO - Calling ComfyUI handler for job clxxxx123
2025-12-13 10:32:15 - local-worker - INFO - Job clxxxx123 completed successfully
2025-12-13 10:32:15 - local-worker - INFO - Successfully notified completion for task clxxxx123
```

## Error Handling

- **Network errors**: Retries with exponential backoff
- **ComfyUI errors**: Captured and sent to callback
- **Processing errors**: Logged and reported as failed jobs
- **Webhook failures**: Multiple retry attempts with backoff

## Systemd Service (Linux)

Create `/etc/systemd/system/gmanimato-worker.service`:

```ini
[Unit]
Description=GmAnimato Local Worker
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/GmComfy
Environment="GMANIMATO_API_URL=http://your-api-url"
Environment="BUCKET_ENDPOINT_URL=https://your-s3-endpoint"
Environment="BUCKET_NAME=your-bucket"
ExecStart=/usr/bin/python3 local_worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable gmanimato-worker
sudo systemctl start gmanimato-worker
sudo systemctl status gmanimato-worker
```

## Monitoring

Check worker logs:

```bash
# View logs (systemd)
sudo journalctl -u gmanimato-worker -f

# View logs (direct run)
python local_worker.py 2>&1 | tee worker.log
```

## Troubleshooting

### Worker not receiving tasks

- Check API URL is correct
- Verify network connectivity to API
- Check that local jobs are being created (queue length < threshold)
- Look for database connection errors in API logs

### Jobs failing immediately

- Verify ComfyUI is running (`http://127.0.0.1:8188`)
- Check ComfyUI has required models
- Review ComfyUI logs for errors
- Enable debug logging: `--debug`

### Webhook failures

- Check callback URL is reachable from worker
- Verify webhook endpoint is responding correctly
- Increase max retries if network is unstable

## Development

```bash
# Run with debug logging
python local_worker.py --debug

# Test task fetching
curl http://localhost:5173/api/worker/task

# Monitor ComfyUI
curl http://127.0.0.1:8188/system_stats
```

## Security Notes

- Worker should run in a trusted environment
- Protect S3 credentials and API URLs
- Consider running worker behind firewall
- Use HTTPS for production API URLs
