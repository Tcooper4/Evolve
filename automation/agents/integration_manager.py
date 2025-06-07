import openai
import logging
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import asyncio
from datetime import datetime
import aiohttp
import os
import subprocess
import tempfile
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class IntegrationManager:
    def __init__(self, config: Dict):
        self.config = config
        self.openai_config = config.get('openai', {})
        self.cursor_config = config.get('cursor', {})
        self.setup_logging()
        self.setup_openai()
        self.setup_cursor()
        self.file_observer = None
        self.active_tasks = {}
        self.code_context = {}

    def setup_logging(self):
        """Configure logging for the integration manager."""
        log_path = Path("automation/logs/integration")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "integration.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_openai(self):
        """Initialize OpenAI configuration."""
        openai.api_key = self.openai_config.get('api_key')
        self.model = self.openai_config.get('model', 'gpt-4')
        self.temperature = self.openai_config.get('temperature', 0.7)
        self.max_tokens = self.openai_config.get('max_tokens', 2000)

    def setup_cursor(self):
        """Initialize Cursor configuration."""
        self.workspace_path = self.cursor_config.get('workspace_path', os.getcwd())
        self.start_file_observer()

    def start_file_observer(self):
        """Start watching for file changes in Cursor workspace."""
        event_handler = CursorFileHandler(self)
        self.file_observer = Observer()
        self.file_observer.schedule(event_handler, self.workspace_path, recursive=True)
        self.file_observer.start()

    async def process_task(self, task: Dict) -> Dict:
        """
        Process a task using OpenAI and Cursor.
        
        Args:
            task: Task specification including type, description, and requirements
        
        Returns:
            Dict: Task results and status
        """
        task_id = task.get('id', str(datetime.now().timestamp()))
        self.active_tasks[task_id] = {
            'status': 'processing',
            'start_time': datetime.now().isoformat(),
            'progress': 0
        }

        try:
            # Step 1: Understand requirements using OpenAI
            requirements = await self._analyze_requirements(task)
            self._update_task_progress(task_id, 20)

            # Step 2: Generate code using OpenAI
            code_changes = await self._generate_code(requirements)
            self._update_task_progress(task_id, 40)

            # Step 3: Apply changes in Cursor
            applied_changes = await self._apply_changes(code_changes)
            self._update_task_progress(task_id, 60)

            # Step 4: Generate tests using OpenAI
            tests = await self._generate_tests(applied_changes)
            self._update_task_progress(task_id, 80)

            # Step 5: Apply tests in Cursor
            test_results = await self._apply_tests(tests)
            self._update_task_progress(task_id, 100)

            # Update task status
            self.active_tasks[task_id].update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'results': {
                    'requirements': requirements,
                    'code_changes': applied_changes,
                    'tests': test_results
                }
            })

            return self.active_tasks[task_id]

        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            self.active_tasks[task_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            raise

    async def _analyze_requirements(self, task: Dict) -> Dict:
        """Analyze task requirements using OpenAI."""
        prompt = f"""
        Analyze the following task and provide detailed requirements:
        
        Type: {task.get('type')}
        Description: {task.get('description')}
        Requirements: {task.get('requirements', '')}
        
        Provide a structured analysis including:
        1. Technical requirements
        2. Dependencies
        3. Implementation approach
        4. Potential challenges
        5. Success criteria
        """

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return json.loads(response.choices[0].message.content)

    async def _generate_code(self, requirements: Dict) -> Dict:
        """Generate code using OpenAI."""
        prompt = f"""
        Generate code based on the following requirements:
        
        {json.dumps(requirements, indent=2)}
        
        Provide:
        1. File structure
        2. Code for each file
        3. Dependencies
        4. Implementation notes
        """

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return json.loads(response.choices[0].message.content)

    async def _apply_changes(self, code_changes: Dict) -> Dict:
        """Apply code changes in Cursor workspace."""
        applied_changes = {}
        
        for file_path, content in code_changes.get('files', {}).items():
            full_path = os.path.join(self.workspace_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
            
            applied_changes[file_path] = {
                'status': 'applied',
                'timestamp': datetime.now().isoformat()
            }
        
        return applied_changes

    async def _generate_tests(self, code_changes: Dict) -> Dict:
        """Generate tests using OpenAI."""
        prompt = f"""
        Generate tests for the following code changes:
        
        {json.dumps(code_changes, indent=2)}
        
        Provide:
        1. Unit tests
        2. Integration tests
        3. Test data
        4. Test documentation
        """

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return json.loads(response.choices[0].message.content)

    async def _apply_tests(self, tests: Dict) -> Dict:
        """Apply and run tests in Cursor workspace."""
        test_results = {}
        
        for test_file, test_content in tests.get('files', {}).items():
            full_path = os.path.join(self.workspace_path, test_file)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(test_content)
            
            # Run tests
            try:
                result = subprocess.run(
                    ['pytest', full_path],
                    capture_output=True,
                    text=True
                )
                test_results[test_file] = {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'output': result.stdout,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                test_results[test_file] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return test_results

    def _update_task_progress(self, task_id: str, progress: int):
        """Update task progress."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['progress'] = progress

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task."""
        return self.active_tasks.get(task_id)

    def get_all_tasks(self) -> Dict:
        """Get status of all tasks."""
        return self.active_tasks

    def stop(self):
        """Stop the integration manager."""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()

class CursorFileHandler(FileSystemEventHandler):
    def __init__(self, integration_manager: IntegrationManager):
        self.integration_manager = integration_manager

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        self.integration_manager.logger.info(f"File modified: {file_path}")
        
        # Update code context
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                self.integration_manager.code_context[file_path] = {
                    'content': content,
                    'last_modified': datetime.now().isoformat()
                }
        except Exception as e:
            self.integration_manager.logger.error(f"Error reading file {file_path}: {str(e)}")

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        self.integration_manager.logger.info(f"File created: {file_path}")

    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        self.integration_manager.logger.info(f"File deleted: {file_path}")
        
        # Remove from code context
        self.integration_manager.code_context.pop(file_path, None) 