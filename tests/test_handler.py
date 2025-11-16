import unittest
from unittest.mock import patch, MagicMock, mock_open, Mock
import sys
import os
import json
import base64
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
    @patch("handler.upload_image")
    @patch.dict(
        os.environ, {"COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES}
    )
    def test_bucket_endpoint_not_configured(self, mock_upload_image, mock_exists):
        mock_exists.return_value = True
        mock_upload_image.return_value = "simulated_uploaded/image.png"
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
    @patch("handler.upload_image")
    @patch.dict(
        os.environ,
        {
            "COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES,
            "BUCKET_ENDPOINT_URL": "http://example.com",
        },
    )
    def test_bucket_endpoint_configured(self, mock_upload_image, mock_exists):
        # Mock the os.path.exists to return True, simulating that the image exists
        mock_exists.return_value = True

        # Mock the rp_upload.upload_image to return a simulated URL
        mock_upload_image.return_value = "http://example.com/uploaded/image.png"

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

    @patch("handler.os.path.exists")
    @patch("handler.upload_image")
    @patch.dict(
        os.environ,
        {
            "COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES,
            "BUCKET_ENDPOINT_URL": "http://example.com",
            "BUCKET_ACCESS_KEY_ID": "",
            "BUCKET_SECRET_ACCESS_KEY": "",
        },
    )
    def test_bucket_image_upload_fails_env_vars_wrong_or_missing(
        self, mock_upload_image, mock_exists
    ):
        # Simulate the file existing in the output path
        mock_exists.return_value = True

        # When AWS credentials are wrong or missing, upload_image should return 'simulated_uploaded/...'
        mock_upload_image.return_value = "simulated_uploaded/image.png"

        outputs = {
            "node_id": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": ""}]}
        }
        job_id = "123"
        with patch("handler.get_file_data", return_value=b"filebytes"):
            files, errors = handler.process_output_files(outputs, job_id)

            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["type"], "s3_url")
            self.assertIn("simulated_uploaded", files[0]["data"])

    @patch("handler.requests.post")
    def test_upload_images_successful(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.text = "Successfully uploaded"
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")

        images = [{"name": "test_image.png", "image": test_image_data}]
        responses = handler.upload_files(images)

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
        responses = handler.upload_files(images)

        self.assertEqual(len(responses), 3)
        self.assertEqual(responses["status"], "error")
