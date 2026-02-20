import runpod
import json
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket
import traceback
import boto3
from botocore.config import Config as BConfig
import random
import threading

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = int(os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 500))
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Websocket reconnection behaviour (can be overridden through environment variables)
# NOTE: more attempts and diagnostics improve debuggability whenever ComfyUI crashes mid-job.
#   • WEBSOCKET_RECONNECT_ATTEMPTS sets how many times we will try to reconnect.
#   • WEBSOCKET_RECONNECT_DELAY_S sets the sleep in seconds between attempts.
#
# If the respective env-vars are not supplied we fall back to sensible defaults ("5" and "3").
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

# Extra verbose websocket trace logs (set WEBSOCKET_TRACE=true to enable)
if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    # This prints low-level frame information to stdout which is invaluable for diagnosing
    # protocol errors but can be noisy in production – therefore gated behind an env-var.
    websocket.enableTrace(True)

# Host where ComfyUI is running
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Helper: quick reachability probe of ComfyUI HTTP endpoint (port 8188)
# ---------------------------------------------------------------------------


def _comfy_server_status():
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
        }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    """
    Attempts to reconnect to the WebSocket server after a disconnect.

    Args:
        ws_url (str): The WebSocket URL (including client_id).
        max_attempts (int): Maximum number of reconnection attempts.
        delay_s (int): Delay in seconds between attempts.
        initial_error (Exception): The error that triggered the reconnect attempt.

    Returns:
        websocket.WebSocket: The newly connected WebSocket object.

    Raises:
        websocket.WebSocketConnectionClosedException: If reconnection fails after all attempts.
    """
    print(
        f"worker-comfyui - Websocket connection closed unexpectedly: {initial_error}. Attempting to reconnect..."
    )
    last_reconnect_error = initial_error
    for attempt in range(max_attempts):
        # Log current server status before each reconnect attempt so that we can
        # see whether ComfyUI is still alive (HTTP port 8188 responding) even if
        # the websocket dropped. This is extremely useful to differentiate
        # between a network glitch and an outright ComfyUI crash/OOM-kill.
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            # If ComfyUI itself is down there is no point in retrying the websocket –
            # bail out immediately so the caller gets a clear "ComfyUI crashed" error.
            print(
                f"worker-comfyui - ComfyUI HTTP unreachable – aborting websocket reconnect: {srv_status.get('error', 'status '+str(srv_status.get('status_code')))}"
            )
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP unreachable during websocket reconnect"
            )

        # Otherwise we proceed with reconnect attempts while server is up
        print(
            f"worker-comfyui - Reconnect attempt {attempt + 1}/{max_attempts}... (ComfyUI HTTP reachable, status {srv_status.get('status_code')})"
        )
        try:
            # Need to create a new socket object for reconnect
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)  # Use existing ws_url
            print(f"worker-comfyui - Websocket reconnected successfully.")
            return new_ws  # Return the new connected socket
        except (
            websocket.WebSocketException,
            ConnectionRefusedError,
            socket.timeout,
            OSError,
        ) as reconn_err:
            last_reconnect_error = reconn_err
            print(
                f"worker-comfyui - Reconnect attempt {attempt + 1} failed: {reconn_err}"
            )
            if attempt < max_attempts - 1:
                print(
                    f"worker-comfyui - Waiting {delay_s} seconds before next attempt..."
                )
                time.sleep(delay_s)
            else:
                print(f"worker-comfyui - Max reconnection attempts reached.")

    # If loop completes without returning, raise an exception
    print("worker-comfyui - Failed to reconnect websocket after connection closed.")
    raise websocket.WebSocketConnectionClosedException(
        f"Connection closed and failed to reconnect. Last error: {last_reconnect_error}"
    )


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate. May contain:
                          - 'workflow': Required workflow definition
                          - 'images': Optional list of images (base64 encoded or URLs)
                          - 'comfy_org_api_key': Optional API key for Comfy.org API Nodes
                          - 'callback_url': Optional webhook URL for job completion notification

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    # Optional: API key for Comfy.org API Nodes, passed per-request
    comfy_org_api_key = job_input.get("comfy_org_api_key")
    
    # Optional: Node weights for progress calculation
    node_weights = job_input.get("node_weights")

    # Build validated response. Only include comfy_org_api_key if it was provided
    validated = {"workflow": workflow, "images": images}
    if comfy_org_api_key is not None:
        validated["comfy_org_api_key"] = comfy_org_api_key
    if node_weights is not None and isinstance(node_weights, dict):
        validated["node_weights"] = node_weights

    callback_url = job_input.get("callback_url")
    if isinstance(callback_url, str) and callback_url.strip() != "":
        validated["callback_url"] = callback_url

    return validated, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    print(f"worker-comfyui - Checking API server at {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException as e:
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"worker-comfyui - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_input_files(images):
    """
    Upload a list of images (base64 encoded or URLs) to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing:
                       - 'name': The filename for the image
                       - 'image': Either a base64 encoded string or a URL to the image

    Returns:
        dict: A dictionary indicating success or error.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")

    proxies = {
        "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
        "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
    }
    # Remove empty entries so requests doesn't error on None values
    proxies = {k: v for k, v in proxies.items() if v}

    for image in images:
        try:
            name = image["name"]
            image_data = image["image"]  # Get the full string (URL or base64)

            # Determine if this is a URL or base64 data
            is_url = image_data.startswith(("http://", "https://"))

            if is_url:
                # Handle URL-based image
                response = requests.get(
                    image_data,
                    timeout=30,
                    proxies=proxies if proxies else None,
                )
                response.raise_for_status()
                blob = response.content
            else:
                # Handle base64 encoded image
                # --- Strip Data URI prefix if present ---
                if "," in image_data:
                    # Find the comma and take everything after it
                    base64_data = image_data.split(",", 1)[1]
                else:
                    # Assume it's already pure base64
                    base64_data = image_data
                # --- End strip ---

                blob = base64.b64decode(base64_data)  # Decode the cleaned data

            # Prepare the form data
            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }

            # POST request to upload the image
            response = requests.post(
                f"http://{COMFY_HOST}/upload/image", files=files, timeout=30
            )
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")

        except base64.binascii.Error as e:
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.Timeout:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.RequestException as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error uploading {image.get('name', 'unknown')}: {e}"
            )
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        print(f"worker-comfyui - image(s) upload finished with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"worker-comfyui - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }
    


def get_available_models():
    """
    Get list of available models from ComfyUI

    Returns:
        dict: Dictionary containing available models by type
    """
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()

        # Extract available checkpoints from CheckpointLoaderSimple
        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = (
                        ckpt_options[0] if isinstance(ckpt_options[0], list) else []
                    )

        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


def queue_workflow(workflow, client_id=None, comfy_org_api_key=None):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed
        client_id (str): The client ID for the websocket connection
        comfy_org_api_key (str, optional): Comfy.org API key for API Nodes

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow

    Raises:
        ValueError: If the workflow validation fails with detailed error information
    """
    # Make client_id optional for backward compatibility with tests
    if client_id is None:
        client_id = ""

    # Include client_id in the prompt payload
    payload = {"prompt": workflow, "client_id": client_id}

    # Optionally inject Comfy.org API key for API Nodes.
    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}
    data = json.dumps(payload).encode("utf-8")

    # Use requests so tests can patch `handler.requests.post`.
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30)

    # Handle validation errors with detailed information (best-effort)
    if response.status_code == 400:
        try:
            error_data = response.json()
            error_message = "Workflow validation failed"
            if isinstance(error_data, dict) and error_data.get("type") == "prompt_outputs_failed_validation":
                raise ValueError(error_data.get("message", error_message))
            raise ValueError(error_message)
        except (json.JSONDecodeError, ValueError):
            raise ValueError(f"ComfyUI validation failed: {response.text}")

    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    # Use requests for consistency and timeout
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_file_data(filename, subfolder, file_type):
    """
    Fetch file bytes from the ComfyUI /view endpoint.

    Args:
        filename (str): The filename of the file.
        subfolder (str): The subfolder where the file is stored.
        file_type (str): The type of the file (e.g., 'output').

    Returns:
        bytes: The raw file data, or None if an error occurs.
    """
    print(
        f"worker-comfyui - Fetching file data: type={file_type}, subfolder={subfolder}, filename={filename}"
    )
    data = {"filename": filename, "subfolder": subfolder, "type": file_type}
    url_values = urllib.parse.urlencode(data)
    try:
        # Use requests for consistency and timeout
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=60)
        response.raise_for_status()
        return response.content
    except requests.Timeout:
        print(f"worker-comfyui - Timeout fetching file data for {filename}")
        return None
    except requests.RequestException as e:
        print(f"worker-comfyui - Error fetching file data for {filename}: {e}")
        return None
    except Exception as e:
        print(
            f"worker-comfyui - Unexpected error fetching file data for {filename}: {e}"
        )
        return None

def process_output_files(outputs, job_id):
    """
    Process outputs (as returned by `get_history`) and either upload files to
    configured bucket or return base64-encoded results. This function is
    agnostic to output keys such as 'images', 'files' or 'gifs' and handles all
    of them using the same logic.

    Returns:
        tuple: (list_of_files, list_of_errors)
            where each file dict contains: 'filename', 'type' ('s3_url'|'base64'), 'data'
    """
    comfy_out = os.environ.get("COMFY_OUTPUT_PATH", "./")
    file_keys = ("images", "files", "gifs")

    output_files = []
    errors = []

    for node_id, node_output in outputs.items():
        for key in file_keys:
            if key not in node_output:
                continue
            for entry in node_output.get(key, []):
                filename = entry.get("filename")
                subfolder = entry.get("subfolder", "")
                ftype = entry.get("type", key)

                if not filename:
                    warn_msg = f"Skipping file in node {node_id} due to missing filename: {entry}"
                    print(f"worker-comfyui - {warn_msg}")
                    errors.append(warn_msg)
                    continue

                # Try to fetch bytes from the /view endpoint first
                file_bytes = get_file_data(filename, subfolder, ftype)

                # Fallback to local file path if /view did not return bytes
                if not file_bytes:
                    local_path = (
                        os.path.join(comfy_out, subfolder, filename)
                        if subfolder
                        else os.path.join(comfy_out, filename)
                    )
                    if os.path.exists(local_path):
                        try:
                            with open(local_path, "rb") as f:
                                file_bytes = f.read()
                        except Exception as e:
                            err = f"Error reading local file {local_path}: {e}"
                            print(f"worker-comfyui - {err}")
                            errors.append(err)
                            continue
                    else:
                        err = f"Failed to fetch file data for {filename} (node {node_id})"
                        print(f"worker-comfyui - {err}")
                        errors.append(err)
                        continue

                # Now we have bytes; either upload to S3 or return as base64
                file_extension = os.path.splitext(filename)[1] or ""

                if os.environ.get("BUCKET_ENDPOINT_URL"):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                            temp_file.write(file_bytes)
                            temp_file_path = temp_file.name

                        s3_url = upload_output_files(job_id, temp_file_path)
                        try:
                            os.remove(temp_file_path)
                        except Exception:
                            pass

                        output_files.append({"filename": filename, "type": "s3_url", "data": s3_url})
                    except Exception as e:
                        err = f"Error uploading {filename} to S3: {e}"
                        print(f"worker-comfyui - {err}")
                        errors.append(err)
                        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                            try:
                                os.remove(temp_file_path)
                            except Exception:
                                pass
                else:
                    try:
                        base64_file = base64.b64encode(file_bytes).decode("utf-8")
                        output_files.append({"filename": filename, "type": "base64", "data": base64_file})
                    except Exception as e:
                        err = f"Error encoding {filename} to base64: {e}"
                        print(f"worker-comfyui - {err}")
                        errors.append(err)

        # Check for unhandled output keys (for debugging)
        other_keys = [k for k in node_output.keys() if k not in file_keys]
        if other_keys:
            warn_msg = f"Node {node_id} produced unhandled output keys: {other_keys}."
            print(f"worker-comfyui - WARNING: {warn_msg}")

    return output_files, errors


def upload_output_files(job_id, file_path):
    """
    Upload a local file to the configured S3-compatible bucket and return the public URL.

    Uses boto3 with a botocore `Config(request_checksum_calculation='when_required')`.
    Requires BUCKET_ENDPOINT_URL and BUCKET_NAME to be configured.

    Args:
        job_id (str): Job identifier, used to build the remote/object key.
        file_path (str): Path to the local file to upload.

    Returns:
        str: URL to the uploaded object.

    Raises:
        Exception: If upload fails or required bucket configuration is missing.
    """
    endpoint = os.environ.get("BUCKET_ENDPOINT_URL")
    bucket = os.environ.get("BUCKET_NAME")
    access_key = os.environ.get("BUCKET_ACCESS_KEY_ID")
    secret_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
    region = os.environ.get("BUCKET_REGION") or None

    if not endpoint or not bucket:
        raise ValueError(
            "S3 upload requires BUCKET_ENDPOINT_URL and BUCKET_NAME to be configured"
        )

    filename = os.path.basename(file_path)
    key = f"{job_id}/{filename}"

    # Use botocore Config to enable request checksum calculation when required
    try:
        client_config = BConfig(request_checksum_calculation="when_required")
    except TypeError:
        client_config = None

    s3_client_kwargs = {
        "endpoint_url": endpoint,
    }
    if access_key is not None:
        s3_client_kwargs["aws_access_key_id"] = access_key
    if secret_key is not None:
        s3_client_kwargs["aws_secret_access_key"] = secret_key
    if region:
        s3_client_kwargs["region_name"] = region
    if client_config is not None:
        s3_client_kwargs["config"] = client_config

    s3 = boto3.client("s3", **s3_client_kwargs)

    # Upload the file
    s3.upload_file(file_path, bucket, key)

    # Construct a public URL (best-effort; exact URL form depends on provider)
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]
    url = f"{endpoint}/{bucket}/{key}"
    return url

def handler(job):
    """
    Handles a job using ComfyUI via websockets for status and image retrieval.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]
    job_id = job["id"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        # Validation failed; skip webhook because callback URL is untrusted
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")
    node_weights = validated_data.get("node_weights")

    # Webhook helper using validated callback_url only
    webhook_url = validated_data.get("callback_url") if isinstance(validated_data, dict) else None

    def _send_webhook_notification(job_id, result, url=webhook_url, async_mode=False):
        """
        Send webhook notification. If async_mode=True, runs retry logic in a background thread
        to avoid blocking the handler from returning and processing the next job.
        
        Args:
            job_id: The job identifier
            result: Result dictionary containing status, errors, files, etc.
            url: Webhook URL to send notification to
            async_mode: If True, runs final notifications in background thread (default: False)
        """
        if not url:
            return False
        
        headers = {"Content-Type": "application/json"}

        payload = {"id": job_id}
        # Choose a status for the webhook based on result
        if result.get("error"):
            payload["status"] = "failed"
            payload["error"] = result.get("error")
            if "details" in result:
                payload["details"] = result.get("details")
        else:
            payload["status"] = result.get("status", "completed")

        # Attach progress information for processing status
        if "progress" in result and isinstance(result["progress"], dict):
            payload["progress"] = result["progress"]

        # Attach discovered files (s3 URLs or base64 blobs) for consumer convenience
        if "files" in result and isinstance(result["files"], list):
            payload["files"] = result["files"]

        # Progress notifications don't retry (fire-and-forget)
        is_progress = result.get("status") == "processing"
        if is_progress:
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if 200 <= resp.status_code < 300:
                    return True
                else:
                    print(f"worker-comfyui - Progress webhook {url} responded with status {resp.status_code}.")
                    return False
            except Exception as e:
                print(f"worker-comfyui - Error sending progress webhook {url}: {e}")
                return False

        # Final status/error notifications use enhanced retry logic with exponential backoff + jitter
        def _retry_webhook():
            print(f"worker-comfyui - Sending webhook notification to {url}.")
            retries = int(os.environ.get("I2V_WEBHOOK_RETRIES", 5))
            base_backoff = float(os.environ.get("I2V_WEBHOOK_BACKOFF_S", 2.0))
            max_backoff = float(os.environ.get("I2V_WEBHOOK_MAX_BACKOFF_S", 60.0))
            timeout = int(os.environ.get("I2V_WEBHOOK_TIMEOUT_S", 30))

            for attempt in range(1, retries + 1):
                try:
                    print(f"worker-comfyui - Sending webhook notification to {url} (attempt {attempt}/{retries})")
                    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                    if 200 <= resp.status_code < 300:
                        print(f"worker-comfyui - Webhook delivered successfully (status {resp.status_code}).")
                        return True
                    else:
                        print(f"worker-comfyui - Webhook responded with status {resp.status_code}: {resp.text}")
                except requests.exceptions.Timeout as e:
                    print(f"worker-comfyui - Webhook timeout (attempt {attempt}): {e}")
                except requests.exceptions.ConnectionError as e:
                    print(f"worker-comfyui - Webhook connection error (attempt {attempt}): {e}")
                except requests.exceptions.RequestException as e:
                    print(f"worker-comfyui - Webhook request error (attempt {attempt}): {e}")
                except Exception as e:
                    print(f"worker-comfyui - Unexpected error sending webhook (attempt {attempt}): {e}")

                if attempt < retries:
                    # Exponential backoff with jitter: base_backoff * 2^(attempt-1) + random jitter
                    exponential_delay = min(base_backoff * (2 ** (attempt - 1)), max_backoff)
                    jitter = random.uniform(0, exponential_delay * 0.1)  # 10% jitter
                    total_delay = exponential_delay + jitter
                    print(f"worker-comfyui - Waiting {total_delay:.2f}s before retry...")
                    time.sleep(total_delay)

            print("worker-comfyui - All webhook attempts failed.")
            return False
        
        # If async_mode is enabled, run webhook retries in background thread
        if async_mode:
            thread = threading.Thread(
                target=_retry_webhook,
                name=f"webhook-{job_id[:8]}",
                daemon=True  # Daemon thread won't prevent process exit
            )
            thread.start()
            print(f"worker-comfyui - Webhook notification started in background thread.")
            return True  # Return immediately, don't wait for thread
        else:
            # Synchronous mode: run retry logic in main thread (for error webhooks)
            return _retry_webhook()

    # Make sure that the ComfyUI HTTP API is available before proceeding
    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        err_result = {
            "error": f"ComfyUI server ({COMFY_HOST}) not reachable after multiple retries."
        }
        try:
            _send_webhook_notification(job_id, err_result)
        except Exception:
            pass
        return err_result

    # Upload input images if they exist
    if input_images:
        upload_result = upload_input_files(input_images)
        if upload_result["status"] == "error":
            # Return upload errors
            err_result = {
                "error": "Failed to upload one or more input images",
                "details": upload_result["details"],
            }
            try:
                _send_webhook_notification(job_id, err_result)
            except Exception:
                pass
            return err_result

    ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_files = []
    errors = []

    try:
        # Establish WebSocket connection
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-comfyui - Connecting to websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-comfyui - Websocket connected")

        # Queue the workflow
        try:
            # Pass per-request API key if provided in input
            queued_workflow = queue_workflow(
                workflow,
                client_id,
                comfy_org_api_key=validated_data.get("comfy_org_api_key"),
            )
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(
                    f"Missing 'prompt_id' in queue response: {queued_workflow}"
                )
            print(f"worker-comfyui - Queued workflow with ID: {prompt_id}")
        except requests.RequestException as e:
            print(f"worker-comfyui - Error queuing workflow: {e}")
            raise ValueError(f"Error queuing workflow: {e}")
        except Exception as e:
            print(f"worker-comfyui - Unexpected error queuing workflow: {e}")
            # For ValueError exceptions from queue_workflow, pass through the original message
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Unexpected error queuing workflow: {e}")

        # Wait for execution completion via WebSocket
        # Use node weights from input (provided by server) or fallback to equal weights
        # Weights reflect actual compute cost of each node for accurate progress
        if node_weights is None:
            node_weights = {}
        default_weight = 1.0  # Light nodes (loaders, preprocessors, etc.)
        
        # Calculate total weight
        total_weight = sum(node_weights.get(node_id, default_weight) for node_id in workflow.keys())
        print(f"worker-comfyui - Using weighted progress with {len(node_weights)} custom weights (total weight: {total_weight:.1f})")
        
        # Track overall workflow progress with weights
        completed_weight = 0.0
        current_node_id = None
        current_node_weight = 0.0
        current_node_progress = 0.0
        executing_nodes = set()
        execution_done = False
        node_progress_count = {}  # Track how many progress events each node has sent
        
        # Progress webhook throttling
        last_progress_webhook = time.time()
        progress_webhook_interval = 2.0  # seconds
        last_progress_check = time.time()
        progress_check_interval = 1.0  # Check and report progress every second for nodes without progress events
        
        def calculate_and_report_progress(force=False):
            """Calculate overall progress and send webhook if interval elapsed."""
            nonlocal last_progress_webhook
            
            # Calculate overall workflow progress using weighted approach
            if total_weight > 0:
                current_progress_weight = completed_weight + (current_node_weight * current_node_progress)
                overall_progress = (current_progress_weight / total_weight) * 100
            else:
                overall_progress = 0
            
            # Also calculate simple node-count-based progress as a fallback indicator
            node_count_progress = (len(executing_nodes) / len(workflow)) * 100 if len(workflow) > 0 else 0
            
            # Send webhook if interval elapsed or forced
            current_time = time.time()
            if force or (current_time - last_progress_webhook >= progress_webhook_interval):
                try:
                    progress_result = {
                        "status": "processing",
                        "progress": {
                            "percentage": round(overall_progress, 1),
                            "completed_nodes": len(executing_nodes) - 1,  # Subtract 1 because current node is in executing_nodes but not completed
                            "total_nodes": len(workflow),
                            "current_node": current_node_id,
                            "current_node_progress": round(current_node_progress * 100, 1)
                        }
                    }
                    _send_webhook_notification(job_id, progress_result)
                    last_progress_webhook = current_time
                except Exception as e:
                    print(f"worker-comfyui - Error sending progress webhook: {e}")
        
        while True:
            try:
                # Check if we should report progress based on time (for nodes that don't emit progress events)
                current_time = time.time()
                if current_node_id and (current_time - last_progress_check >= progress_check_interval):
                    calculate_and_report_progress()
                    last_progress_check = current_time
                
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "status":
                        status_data = message.get("data", {}).get("status", {})
                    elif message.get("type") == "executing":
                        # Track node execution for overall progress
                        data = message.get("data", {})
                        node_id = data.get("node")
                        
                        if node_id is None and data.get("prompt_id") == prompt_id:
                            # Execution finished
                            print(f"worker-comfyui - Execution finished for prompt {prompt_id}")
                            execution_done = True
                            break
                        elif node_id is not None:
                            # New node started executing
                            if node_id not in executing_nodes:
                                executing_nodes.add(node_id)
                                if current_node_id is not None:
                                    # Previous node completed - add its full weight
                                    completed_weight += current_node_weight
                                
                                # Set up new current node
                                current_node_id = node_id
                                current_node_weight = node_weights.get(node_id, default_weight)
                                current_node_progress = 0.0
                                
                                node_desc = " (HEAVY)" if current_node_weight > 10 else ""
                                
                                # Report progress when a new node starts (previous node completed)
                                calculate_and_report_progress()
                    elif message.get("type") == "progress":
                        data = message.get("data", {})
                        value = data.get("value", 0)
                        max_val = data.get("max", 0)
                        node = data.get("node")
                        
                        if max_val > 0:
                            # Make sure we have the weight for the current node from the progress message
                            # This handles cases where progress arrives before the executing message
                            if node and (current_node_id != node):
                                current_node_id = node
                                current_node_weight = node_weights.get(node, default_weight)
                            
                            # Track progress event count for this node
                            if node:
                                node_progress_count[node] = node_progress_count.get(node, 0) + 1
                            
                            # Calculate progress (0.0 to 1.0)
                            reported_progress = value / max_val
                            
                            # Some nodes weirdly report 100% on first message, then proper values
                            # Ignore the first progress event if it reports >= 99% completion
                            if node and node_progress_count.get(node, 0) == 1 and reported_progress >= 0.99:
                                print(f"worker-comfyui - Node {node} reported {reported_progress * 100:.1f}% on first progress event, ignoring (likely spurious)")
                            else:
                                # Update current node progress
                                current_node_progress = reported_progress
                                
                                # Report progress using the helper function
                                calculate_and_report_progress()
                    elif message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node Type: {data.get('node_type')}, Node ID: {data.get('node_id')}, Message: {data.get('exception_message')}"
                            print(
                                f"worker-comfyui - Execution error received: {error_details}"
                            )
                            errors.append(f"Workflow execution error: {error_details}")
                            break
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                print(f"worker-comfyui - Websocket receive timed out. Still waiting...")
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    # Attempt to reconnect
                    ws = _attempt_websocket_reconnect(
                        ws_url,
                        WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S,
                        closed_err,
                    )

                    print(
                        "worker-comfyui - Resuming message listening after successful reconnect."
                    )
                    continue
                except (
                    websocket.WebSocketConnectionClosedException
                ) as reconn_failed_err:
                    # If _attempt_websocket_reconnect fails, it raises this exception
                    # Let this exception propagate to the outer handler's except block
                    raise reconn_failed_err

            except json.JSONDecodeError:
                print(f"worker-comfyui - Received invalid JSON message via websocket.")

        if not execution_done and not errors:
            raise ValueError(
                "Workflow monitoring loop exited without confirmation of completion or error."
            )

        # Fetch history even if there were execution errors, some outputs might exist
        print(f"worker-comfyui - Fetching history for prompt {prompt_id}...")
        history = get_history(prompt_id)

        if prompt_id not in history:
            error_msg = f"Prompt ID {prompt_id} not found in history after execution."
            print(f"worker-comfyui - {error_msg}")
            if not errors:
                err_result = {"error": error_msg}
                try:
                    _send_webhook_notification(job_id, err_result)
                except Exception:
                    pass
                return err_result
            else:
                errors.append(error_msg)
                err_result = {
                    "error": "Job processing failed, prompt ID not found in history.",
                    "details": errors,
                }
                try:
                    _send_webhook_notification(job_id, err_result)
                except Exception:
                    pass
                return err_result

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})

        if not outputs:
            warning_msg = f"No outputs found in history for prompt {prompt_id}."
            print(f"worker-comfyui - {warning_msg}")
            if not errors:
                errors.append(warning_msg)

        print(f"worker-comfyui - Processing outputs for prompt {prompt_id}...")
        files_from_nodes, proc_errors = process_output_files(outputs, job_id)
        if files_from_nodes:
            output_files.extend(files_from_nodes)
        if proc_errors:
            errors.extend(proc_errors)

    except websocket.WebSocketException as e:
        print(f"worker-comfyui - WebSocket Error: {e}")
        print(traceback.format_exc())
        err_result = {"error": f"WebSocket communication error: {e}"}
        try:
            _send_webhook_notification(job_id, err_result)
        except Exception:
            pass
        return err_result
    except requests.RequestException as e:
        print(f"worker-comfyui - HTTP Request Error: {e}")
        print(traceback.format_exc())
        err_result = {"error": f"HTTP communication error with ComfyUI: {e}"}
        try:
            _send_webhook_notification(job_id, err_result)
        except Exception:
            pass
        return err_result
    except ValueError as e:
        print(f"worker-comfyui - Value Error: {e}")
        print(traceback.format_exc())
        err_result = {"error": str(e)}
        try:
            _send_webhook_notification(job_id, err_result)
        except Exception:
            pass
        return err_result
    except Exception as e:
        print(f"worker-comfyui - Unexpected Handler Error: {e}")
        print(traceback.format_exc())
        err_result = {"error": f"An unexpected error occurred: {e}"}
        try:
            _send_webhook_notification(job_id, err_result)
        except Exception:
            pass
        return err_result
    finally:
        if ws and ws.connected:
            print(f"worker-comfyui - Closing websocket connection.")
            ws.close()

    final_result = {}

    if output_files:
        final_result["files"] = output_files

    if errors:
        final_result["errors"] = errors
        print(f"worker-comfyui - Job completed with errors/warnings: {errors}")

    # If there are no outputs at all and there were errors, surface failure
    if not output_files and errors:
        print(f"worker-comfyui - Job failed with no output files.")
        err_result = {"error": "Job processing failed", "details": errors}
        try:
            _send_webhook_notification(job_id, err_result)
        except Exception:
            pass
        return err_result
    # If there are no outputs and no errors, mark success but no files
    elif not output_files and not errors:
        print(
            f"worker-comfyui - Job completed successfully, but the workflow produced no files."
        )
        final_result["status"] = "success_no_files"
        final_result["files"] = []

    total_outputs = len(output_files)
    print(f"worker-comfyui - Job completed. Returning {total_outputs} output(s).")
    try:
        # Best-effort: notify external system about completion in background thread
        # This allows the handler to return immediately and process the next job
        _send_webhook_notification(job_id, final_result, async_mode=True)
    except Exception as e:
        print(f"worker-comfyui - Unexpected error notifying webhook: {e}")

    return final_result


if __name__ == "__main__":
    print("worker-comfyui - Starting handler...")
    runpod.serverless.start({"handler": handler})
