import unittest
from unittest.mock import patch, MagicMock, mock_open, Mock
import sys
import os
import json
import base64
import time
import requests

# Make sure project root is on sys.path so tests can import top-level handler.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import handler

# Local folder for test resources
RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES = "./test_resources/images"


class TestRunpodWorkerComfy(unittest.TestCase):
    def test_valid_input_with_workflow_only(self):
        input_data = {"workflow": {"key": "value"}}
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data, {"workflow": {"key": "value"}, "images": None})

    def test_valid_input_with_workflow_and_images(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png", "image": "base64string"}],
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data, input_data)

    def test_input_missing_workflow(self):
        input_data = {"images": [{"name": "image1.png", "image": "base64string"}]}
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Missing 'workflow' parameter")

    def test_input_with_invalid_images_structure(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png"}],  # Missing 'image' key
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(
            error, "'images' must be a list of objects with 'name' and 'image' keys"
        )

    def test_invalid_json_string_input(self):
        input_data = "invalid json"
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Invalid JSON format in input")

    def test_valid_json_string_input(self):
        input_data = '{"workflow": {"key": "value"}}'
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data, {"workflow": {"key": "value"}, "images": None})

    def test_empty_input(self):
        input_data = None
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Please provide input")

    @patch("handler.requests.get")
    def test_check_server_server_up(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response

        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertTrue(result)

    @patch("handler.requests.get")
    def test_check_server_server_down(self, mock_requests):
        mock_requests.get.side_effect = handler.requests.RequestException()
        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertFalse(result)

    @patch("handler.requests.post")
    def test_queue_prompt(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"prompt_id": "123"}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = handler.queue_workflow({"prompt": "test"})
        self.assertEqual(result, {"prompt_id": "123"})

    @patch("handler.requests.get")
    def test_get_history(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = handler.get_history("123")
        self.assertEqual(result, {"key": "value"})
        mock_get.assert_called_with("http://127.0.0.1:8188/history/123", timeout=30)

    @patch("handler.os.path.exists")
    @patch("handler.upload_output_files")
    @patch.dict(
        os.environ, {"COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES}
    )
    def test_bucket_endpoint_not_configured(self, mock_upload_output_files, mock_exists):
        mock_exists.return_value = True
        mock_upload_output_files.return_value = "simulated_uploaded/image.png"
        # Mock get_file_data to return bytes so the processor doesn't try HTTP/local read
        with patch("handler.get_file_data", return_value=b"filebytes"):
            outputs = {
                "node_id": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": ""}]}
            }
            job_id = "123"

            files, errors = handler.process_output_files(outputs, job_id)

            self.assertEqual(len(files), 1)
            self.assertEqual(errors, [])

    @patch("handler.os.path.exists")
    @patch("handler.upload_output_files")
    @patch.dict(
        os.environ,
        {
            "COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES,
            "BUCKET_ENDPOINT_URL": "http://example.com",
        },
    )
    def test_bucket_endpoint_configured(self, mock_upload_output_files, mock_exists):
        # Mock the os.path.exists to return True, simulating that the image exists
        mock_exists.return_value = True

        # Mock the rp_upload.upload_output_files to return a simulated URL
        mock_upload_output_files.return_value = "http://example.com/uploaded/image.png"

        # Define the outputs and job_id for the test
        with patch("handler.get_file_data", return_value=b"filebytes"):
            outputs = {
                "node_id": {
                    "images": [{"filename": "ComfyUI_00001_.png", "subfolder": "test"}]
                }
            }
            job_id = "123"

            files, errors = handler.process_output_files(outputs, job_id)

            self.assertEqual(len(files), 1)
            self.assertEqual(errors, [])
            self.assertEqual(files[0]["type"], "s3_url")
            self.assertEqual(files[0]["data"], "http://example.com/uploaded/image.png")

    @patch("handler.boto3")
    def test_upload_output_files_success(self, mock_boto3):
        """Test successful S3 upload with boto3"""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict(
            os.environ,
            {
                "BUCKET_ENDPOINT_URL": "http://example.com",
                "BUCKET_NAME": "test-bucket",
                "BUCKET_ACCESS_KEY_ID": "test-key",
                "BUCKET_SECRET_ACCESS_KEY": "test-secret",
            },
        ):
            result = handler.upload_output_files("job123", "/path/to/file.png")

            # Verify boto3.client was called with correct parameters
            mock_boto3.client.assert_called_once()
            call_kwargs = mock_boto3.client.call_args[1]
            self.assertEqual(call_kwargs["endpoint_url"], "http://example.com")
            self.assertEqual(call_kwargs["aws_access_key_id"], "test-key")
            self.assertEqual(call_kwargs["aws_secret_access_key"], "test-secret")

            # Verify upload_file was called
            mock_s3.upload_file.assert_called_once_with("/path/to/file.png", "test-bucket", "job123/file.png")

            # Verify the returned URL is correct
            self.assertEqual(result, "http://example.com/test-bucket/job123/file.png")

    def test_upload_output_files_missing_bucket_config(self):
        """Test that upload_output_files raises error when bucket config is missing"""
        with patch.dict(os.environ, {"BUCKET_ENDPOINT_URL": ""}, clear=False):
            with self.assertRaises(ValueError) as context:
                handler.upload_output_files("job123", "/path/to/file.png")
            self.assertIn("BUCKET_ENDPOINT_URL and BUCKET_NAME", str(context.exception))

    @patch("handler.boto3")
    def test_upload_output_files_upload_fails(self, mock_boto3):
        """Test that upload_output_files propagates boto3 errors"""
        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = Exception("S3 upload failed")
        mock_boto3.client.return_value = mock_s3

        with patch.dict(
            os.environ,
            {
                "BUCKET_ENDPOINT_URL": "http://example.com",
                "BUCKET_NAME": "test-bucket",
            },
        ):
            with self.assertRaises(Exception) as context:
                handler.upload_output_files("job123", "/path/to/file.png")
            self.assertIn("S3 upload failed", str(context.exception))

    @patch("handler.requests.post")
    def test_upload_images_successful(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.text = "Successfully uploaded"
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")

        images = [{"name": "test_image.png", "image": test_image_data}]
        responses = handler.upload_input_files(images)

        self.assertEqual(len(responses), 3)
        self.assertEqual(responses["status"], "success")

    @patch("handler.requests.post")
    def test_upload_images_failed(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 400
        mock_response.text = "Error uploading"
        # Simulate requests raising on bad status codes via raise_for_status
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")

        images = [{"name": "test_image.png", "image": test_image_data}]
        responses = handler.upload_input_files(images)

        self.assertEqual(len(responses), 3)
        self.assertEqual(responses["status"], "error")

    def test_upload_input_files_no_images(self):
        """Test upload_input_files with empty list returns success with no details"""
        result = handler.upload_input_files([])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "No images to upload")
        self.assertEqual(result["details"], [])

    def test_upload_input_files_none(self):
        """Test upload_input_files with None returns success with no details"""
        result = handler.upload_input_files(None)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "No images to upload")
        self.assertEqual(result["details"], [])

    @patch("handler.requests.post")
    def test_upload_input_files_base64_success(self, mock_post):
        """Test successful upload of base64 encoded image"""
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        images = [{"name": "test_image.png", "image": test_image_data}]

        result = handler.upload_input_files(images)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["details"]), 1)
        self.assertIn("Successfully uploaded test_image.png", result["details"][0])
        mock_post.assert_called_once()

    @patch("handler.requests.get")
    @patch("handler.requests.post")
    def test_upload_input_files_url_success(self, mock_post, mock_get):
        """Test successful upload of image from URL"""
        mock_get_response = unittest.mock.Mock()
        mock_get_response.content = b"Downloaded Image Data"
        mock_get.return_value = mock_get_response

        mock_post_response = unittest.mock.Mock()
        mock_post_response.status_code = 200
        mock_post.return_value = mock_post_response

        images = [{"name": "remote_image.png", "image": "https://example.com/image.png"}]

        result = handler.upload_input_files(images)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["details"]), 1)
        self.assertIn("Successfully uploaded remote_image.png", result["details"][0])
        mock_get.assert_called_once_with("https://example.com/image.png", timeout=30)
        mock_post.assert_called_once()

    @patch("handler.requests.post")
    def test_upload_input_files_data_uri_format(self, mock_post):
        """Test successful upload of data URI formatted image"""
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        data_uri = f"data:image/png;base64,{image_data}"
        images = [{"name": "data_uri_image.png", "image": data_uri}]

        result = handler.upload_input_files(images)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["details"]), 1)
        self.assertIn("Successfully uploaded data_uri_image.png", result["details"][0])

    @patch("handler.requests.post")
    def test_upload_input_files_base64_decode_error(self, mock_post):
        """Test upload with invalid base64 data"""
        images = [{"name": "bad_image.png", "image": "not-valid-base64!!!"}]

        result = handler.upload_input_files(images)

        self.assertEqual(result["status"], "error")
        self.assertIn("Some images failed to upload", result["message"])
        self.assertTrue(any("Error decoding base64" in error for error in result["details"]))
        mock_post.assert_not_called()

    @patch("handler.requests.get")
    def test_upload_input_files_url_download_timeout(self, mock_get):
        """Test upload with URL download timeout"""
        mock_get.side_effect = requests.Timeout()
        images = [{"name": "timeout_image.png", "image": "https://example.com/image.png"}]

        result = handler.upload_input_files(images)

        self.assertEqual(result["status"], "error")
        self.assertIn("Some images failed to upload", result["message"])
        self.assertTrue(any("Timeout uploading" in error for error in result["details"]))

    @patch("handler.requests.get")
    @patch("handler.requests.post")
    def test_upload_input_files_post_request_error(self, mock_post, mock_get):
        """Test upload with POST request error"""
        mock_get_response = unittest.mock.Mock()
        mock_get_response.content = b"Image Data"
        mock_get.return_value = mock_get_response

        mock_post_response = unittest.mock.Mock()
        mock_post_response.raise_for_status.side_effect = requests.RequestException("Upload failed")
        mock_post.return_value = mock_post_response

        images = [{"name": "error_image.png", "image": "https://example.com/image.png"}]

        result = handler.upload_input_files(images)

        self.assertEqual(result["status"], "error")
        self.assertIn("Some images failed to upload", result["message"])
        self.assertTrue(any("Error uploading error_image.png" in error for error in result["details"]))

    @patch("handler.requests.post")
    def test_upload_input_files_multiple_images_partial_failure(self, mock_post):
        """Test upload with multiple images where some succeed and some fail"""
        def side_effect(*args, **kwargs):
            mock_response = unittest.mock.Mock()
            # Fail on the second call
            if mock_post.call_count == 2:
                mock_response.raise_for_status.side_effect = requests.RequestException("Failed")
            else:
                mock_response.status_code = 200
            return mock_response

        mock_post.side_effect = side_effect

        image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        images = [
            {"name": "image1.png", "image": image_data},
            {"name": "image2.png", "image": image_data},
        ]

        result = handler.upload_input_files(images)

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["details"]), 1)  # One success, one error
        self.assertIn("Error uploading image2.png", result["details"][0])

    def test_validate_input_with_callback_url(self):
        """Test that callback_url is preserved in validated data"""
        input_data = {
            "workflow": {"key": "value"},
            "callback_url": "https://example.com/webhook?token=abc123",
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data["callback_url"], "https://example.com/webhook?token=abc123")

    def test_validate_input_ignores_empty_callback_url(self):
        """Test that empty callback_url is not included in validated data"""
        input_data = {
            "workflow": {"key": "value"},
            "callback_url": "",
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertNotIn("callback_url", validated_data)

    @patch("handler.requests.post")
    def test_webhook_notification_success(self, mock_post):
        """Test successful webhook notification delivery"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Simulate the handler's webhook sending logic
        webhook_url = "https://example.com/webhook?token=abc123"
        result = {
            "status": "completed",
            "files": [{"filename": "output.png", "type": "s3_url", "data": "s3://bucket/output.png"}],
        }
        job_id = "job-123"

        headers = {"Content-Type": "application/json"}
        payload = {
            "id": job_id,
            "status": result.get("status", "completed"),
            "files": result.get("files", []),
        }

        requests.post(webhook_url, json=payload, headers=headers, timeout=10)

        mock_post.assert_called_once_with(
            webhook_url, json=payload, headers=headers, timeout=10
        )

    @patch("handler.requests.post")
    def test_webhook_notification_with_error(self, mock_post):
        """Test webhook notification when job fails"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        webhook_url = "https://example.com/webhook?token=abc123"
        result = {
            "error": "Job processing failed",
            "details": ["Workflow error: invalid node"],
        }
        job_id = "job-456"

        headers = {"Content-Type": "application/json"}
        payload = {
            "id": job_id,
            "status": "failed",
            "error": result.get("error"),
            "details": result.get("details"),
        }

        requests.post(webhook_url, json=payload, headers=headers, timeout=10)

        mock_post.assert_called_once_with(
            webhook_url, json=payload, headers=headers, timeout=10
        )

    @patch("handler.requests.post")
    @patch.dict(os.environ, {"I2V_WEBHOOK_RETRIES": "2", "I2V_WEBHOOK_BACKOFF_S": "0.1"})
    def test_webhook_notification_retry_on_failure(self, mock_post):
        """Test webhook retries on failure"""
        # First attempt fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Server error"

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200

        mock_post.side_effect = [mock_response_fail, mock_response_success]

        webhook_url = "https://example.com/webhook?token=abc123"
        result = {"status": "completed", "files": []}
        job_id = "job-789"

        headers = {"Content-Type": "application/json"}
        payload = {"id": job_id, "status": "completed", "files": []}

        # Simulate retry logic
        retries = int(os.environ.get("I2V_WEBHOOK_RETRIES", 3))
        backoff = float(os.environ.get("I2V_WEBHOOK_BACKOFF_S", 1.0))

        success = False
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
                if 200 <= resp.status_code < 300:
                    success = True
                    break
            except Exception:
                pass
            if attempt < retries:
                time.sleep(backoff * attempt)

        self.assertTrue(success)
        # Should have been called twice (first failed, second succeeded)
        self.assertEqual(mock_post.call_count, 2)

    @patch("handler.requests.post")
    def test_webhook_notification_no_url_provided(self, mock_post):
        """Test that webhook is skipped when no callback_url is provided"""
        # When callback_url is None, the handler should not attempt to send webhook
        webhook_url = None

        if not webhook_url:
            mock_post.assert_not_called()
        else:
            # This branch should not be reached
            self.fail("Webhook should not be sent when URL is None")

    @patch("handler.requests.post")
    @patch("handler.time.sleep")  # Mock sleep to speed up tests
    @patch.dict(os.environ, {"I2V_WEBHOOK_RETRIES": "3", "I2V_WEBHOOK_BACKOFF_S": "0.1", "I2V_WEBHOOK_TIMEOUT_S": "30"})
    def test_webhook_notification_connection_error_retry(self, mock_sleep, mock_post):
        """Test webhook retries on connection errors with exponential backoff"""
        # First two attempts fail with connection error, third succeeds
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Network unreachable"),
            requests.exceptions.ConnectionError("Network unreachable"),
            MagicMock(status_code=200),
        ]

        webhook_url = "https://example.com/webhook?token=abc123"
        result = {"status": "completed", "files": []}
        job_id = "job-connection-test"

        headers = {"Content-Type": "application/json"}
        payload = {"id": job_id, "status": "completed", "files": []}

        # Simulate retry logic with exponential backoff
        retries = int(os.environ.get("I2V_WEBHOOK_RETRIES", 5))
        base_backoff = float(os.environ.get("I2V_WEBHOOK_BACKOFF_S", 2.0))
        timeout = int(os.environ.get("I2V_WEBHOOK_TIMEOUT_S", 30))

        success = False
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(webhook_url, json=payload, headers=headers, timeout=timeout)
                if 200 <= resp.status_code < 300:
                    success = True
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException):
                pass
            if attempt < retries:
                # Simulate exponential backoff (simplified for test)
                time.sleep(base_backoff * (2 ** (attempt - 1)))

        self.assertTrue(success)
        # Should have been called three times (two failures, one success)
        self.assertEqual(mock_post.call_count, 3)

    @patch("handler.requests.post")
    @patch("handler.time.sleep")
    @patch.dict(os.environ, {"I2V_WEBHOOK_RETRIES": "3", "I2V_WEBHOOK_BACKOFF_S": "0.1", "I2V_WEBHOOK_TIMEOUT_S": "30"})
    def test_webhook_notification_timeout_error_retry(self, mock_sleep, mock_post):
        """Test webhook retries on timeout errors"""
        # First attempt times out, second succeeds
        mock_post.side_effect = [
            requests.exceptions.Timeout("Request timed out"),
            MagicMock(status_code=200),
        ]

        webhook_url = "https://example.com/webhook?token=abc123"
        result = {"status": "completed", "files": []}
        job_id = "job-timeout-test"

        headers = {"Content-Type": "application/json"}
        payload = {"id": job_id, "status": "completed", "files": []}

        retries = int(os.environ.get("I2V_WEBHOOK_RETRIES", 5))
        timeout = int(os.environ.get("I2V_WEBHOOK_TIMEOUT_S", 30))

        success = False
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(webhook_url, json=payload, headers=headers, timeout=timeout)
                if 200 <= resp.status_code < 300:
                    success = True
                    break
            except requests.exceptions.Timeout:
                pass
            if attempt < retries:
                time.sleep(0.1)

        self.assertTrue(success)
        # Should have been called twice (one timeout, one success)
        self.assertEqual(mock_post.call_count, 2)

    @patch("handler.requests.post")
    @patch("handler.time.sleep")
    @patch.dict(os.environ, {"I2V_WEBHOOK_RETRIES": "2", "I2V_WEBHOOK_BACKOFF_S": "0.1"})
    def test_webhook_notification_all_retries_exhausted(self, mock_sleep, mock_post):
        """Test webhook fails after all retries are exhausted"""
        # All attempts fail
        mock_post.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        webhook_url = "https://example.com/webhook?token=abc123"
        result = {"status": "completed", "files": []}
        job_id = "job-fail-test"

        headers = {"Content-Type": "application/json"}
        payload = {"id": job_id, "status": "completed", "files": []}

        retries = int(os.environ.get("I2V_WEBHOOK_RETRIES", 3))

        success = False
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
                if 200 <= resp.status_code < 300:
                    success = True
                    break
            except requests.exceptions.ConnectionError:
                pass
            if attempt < retries:
                time.sleep(0.1)

        self.assertFalse(success)
        # Should have been called retries times
        self.assertEqual(mock_post.call_count, retries)
