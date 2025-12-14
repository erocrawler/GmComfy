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


class LocalWorker:
    """Worker that polls for and processes local jobs"""
    
    def __init__(self, api_url, poll_interval=5, worker_secret: str | None = None):
        self.api_url = api_url.rstrip('/')
        self.poll_interval = poll_interval
        self.worker_secret = worker_secret
        self.task_url = f"{self.api_url}/api/worker/task"
        logger.info(f"Initialized worker with API URL: {self.api_url}")
        logger.info(f"Poll interval: {self.poll_interval}s")
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
        
        # Build ComfyUI job input from task data
        # The task contains: id, job_id, original_image_url, prompt, callback_url, etc.
        job_input = {
            'workflow': task.get('workflow', {}),
            'callback_url': task.get('callback_url'),
        }
        
        # If there's an image URL, add it to the images array
        if task.get('original_image_url'):
            image_filename = f"{job_id}.png"
            job_input['images'] = [{
                'name': image_filename,
                'image': task['original_image_url']
            }]
        
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
            return {'error': f"Processing error: {str(e)}"}
    
    def run(self):
        """Main worker loop"""
        logger.info("Starting local worker...")
        
        while True:
            try:
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
        worker_secret=args.worker_secret
    )
    
    try:
        worker.run()
    except Exception as e:
        logger.critical(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
