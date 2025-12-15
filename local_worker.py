#!/usr/bin/env python3
"""
Local Worker for GmComfy
Polls the Worker API for pending local jobs and processes them using ComfyUI handler.

Motivation:
This worker allows the system to run on a single dedicated pod instead of serverless (for better pricing/performance on light traffic).
"""

import os
import sys
import time
import requests
import argparse
import logging
from handler import handler as comfy_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('local-worker')

# Default configuration
DEFAULT_API_URL = os.environ.get('WORKER_API_URL', 'http://localhost:5173')
DEFAULT_POLL_INTERVAL = int(os.environ.get('WORKER_POLL_INTERVAL', 5))
DEFAULT_WORKER_SECRET = os.environ.get('WORKER_TASK_SECRET')
DEFAULT_SENTINEL_FILE = os.environ.get('WORKER_SENTINEL_FILE', '.worker_stop')


class LocalWorker:
    """Worker that polls for and processes local jobs"""
    
    def __init__(self, api_url, poll_interval=5, worker_secret: str | None = None, sentinel_file: str = DEFAULT_SENTINEL_FILE):
        self.api_url = api_url.rstrip('/')
        self.poll_interval = poll_interval
        self.worker_secret = worker_secret
        self.sentinel_file = sentinel_file
        self.task_url = f"{self.api_url}/api/worker/task"
        logger.info(f"Initialized worker with API URL: {self.api_url}")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info(f"Sentinel file: {self.sentinel_file}")
        if not self.worker_secret:
            logger.warning("No WORKER_TASK_SECRET provided; task endpoint may reject requests")
    
    def fetch_task(self):
        """
        Fetch the next available task from the API
        
        Returns:
            dict: Task data with job details, or None if no task available
        """
        try:
            logger.debug(f"Polling for tasks at {self.task_url}")
            headers = {}
            if self.worker_secret:
                headers['x-worker-secret'] = self.worker_secret
            response = requests.get(self.task_url, headers=headers, timeout=10)
            
            if response.status_code == 404:
                # No tasks available
                logger.debug("No tasks available")
                return None
            
            if response.status_code == 200:
                task = response.json()
                logger.info(f"Received task: {task.get('id')}")
                return task
            
            logger.warning(f"Unexpected status code {response.status_code}: {response.text}")
            return None
            
        except requests.Timeout:
            logger.error("Timeout fetching task from API")
            return None
        except requests.RequestException as e:
            logger.error(f"Error fetching task: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching task: {e}")
            return None
    
    def process_task(self, task):
        """
        Process a task using the ComfyUI handler
        
        Args:
            task (dict): Task data from the API
            
        Returns:
            dict: Processing result
        """
        job_id = task.get('id')
        logger.info(f"Processing task {job_id}")
        
        # The task already contains the complete workflow payload from buildWorkflow()
        # It has structure: { id, job_id, input: { workflow: {...}, images: [...], callback_url: ... } }
        # We just need to pass it to the handler in the right format
        
        job_input = task.get('input')
        if not job_input:
            logger.error(f"Task {job_id} missing 'input' field")
            return {'error': 'Task missing input field'}
        
        # Create a job object that matches the handler's expected format
        job = {
            'id': job_id,
            'input': job_input
        }
        
        try:
            logger.info(f"Calling ComfyUI handler for job {job_id}")
            result = comfy_handler(job)
            
            if result.get('error'):
                logger.error(f"Job {job_id} failed: {result.get('error')}")
            else:
                logger.info(f"Job {job_id} completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
            error_result = {'error': f"Processing error: {str(e)}"}
            
            # Try to send failure notification to webhook
            callback_url = job_input.get('callback_url') if isinstance(job_input, dict) else None
            if callback_url:
                try:
                    import requests
                    headers = {'Content-Type': 'application/json'}
                    payload = {
                        'id': job_id,
                        'status': 'failed',
                        'error': str(e)
                    }
                    requests.post(callback_url, json=payload, headers=headers, timeout=10)
                    logger.info(f"Sent failure notification for job {job_id}")
                except Exception as webhook_err:
                    logger.error(f"Failed to send failure webhook for job {job_id}: {webhook_err}")
            
            return error_result
    
    def check_sentinel(self):
        """Check if sentinel file exists (signal to stop worker)"""
        return os.path.exists(self.sentinel_file)
    
    def run(self):
        """Main worker loop"""
        logger.info("Starting local worker...")
        logger.info(f"To gracefully stop the worker, create file: {self.sentinel_file}")
        
        while True:
            try:
                # Check for sentinel file
                if self.check_sentinel():
                    logger.info(f"Sentinel file '{self.sentinel_file}' detected, shutting down gracefully...")
                    try:
                        os.remove(self.sentinel_file)
                        logger.info(f"Removed sentinel file '{self.sentinel_file}'")
                    except Exception as e:
                        logger.warning(f"Could not remove sentinel file: {e}")
                    break
                
                # Fetch next task
                task = self.fetch_task()
                
                if task:
                    # Process the task
                    # Note: handler.py automatically sends webhook notification to callback_url
                    result = self.process_task(task)
                    
                    # Brief pause before polling again
                    time.sleep(1)
                else:
                    # No tasks available, wait before polling again
                    time.sleep(self.poll_interval)
                    
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(self.poll_interval)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GmComfy Local Worker')
    parser.add_argument(
        '--api-url',
        default=DEFAULT_API_URL,
        help=f'Worker API URL (default: {DEFAULT_API_URL})'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f'Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL})'
    )
    parser.add_argument(
        '--worker-secret',
        default=DEFAULT_WORKER_SECRET,
        help='Shared secret for /api/worker/task authentication (env WORKER_TASK_SECRET)'
    )
    parser.add_argument(
        '--sentinel-file',
        default=DEFAULT_SENTINEL_FILE,
        help=f'Sentinel file path for graceful shutdown (default: {DEFAULT_SENTINEL_FILE})'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create and run worker
    worker = LocalWorker(
        api_url=args.api_url,
        poll_interval=args.poll_interval,
        worker_secret=args.worker_secret,
        sentinel_file=args.sentinel_file
    )
    
    try:
        worker.run()
    except Exception as e:
        logger.critical(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
