import os
import logging
from typing import Dict, List, Optional
import json
from pathlib import Path
import openai
from dotenv import load_dotenv
import subprocess
import shutil
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentAgent:
    def __init__(self, config: Dict):
        """Initialize the deployment agent."""
        self.config = config
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        
        # Load environment variables
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    async def deploy_code(self, code: str, tests: str, task: Dict) -> Dict:
        """Deploy code and tests."""
        try:
            # Create deployment plan
            plan = await self._create_deployment_plan(code, tests, task)
            
            # Execute deployment steps
            results = await self._execute_deployment_plan(plan)
            
            # Verify deployment
            verification = await self._verify_deployment(results)
            
            return {
                "plan": plan,
                "results": results,
                "verification": verification
            }
            
        except Exception as e:
            logger.error(f"Error deploying code: {str(e)}")
            raise

    async def _create_deployment_plan(self, code: str, tests: str, task: Dict) -> Dict:
        """Create a deployment plan."""
        try:
            # Prepare the prompt
            prompt = f"""Create a deployment plan for the following code and tests:
            
            Code:
            {code}
            
            Tests:
            {tests}
            
            Task:
            {json.dumps(task, indent=2)}
            
            Include:
            1. Environment setup
            2. Dependencies installation
            3. Code deployment steps
            4. Test execution
            5. Verification steps
            6. Rollback procedures
            
            Format the response as a JSON object with these categories."""
            
            # Generate plan
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in software deployment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error creating deployment plan: {str(e)}")
            raise

    async def _execute_deployment_plan(self, plan: Dict) -> Dict:
        """Execute the deployment plan."""
        results = {
            "steps": [],
            "success": True,
            "errors": []
        }
        
        try:
            # Execute each step in the plan
            for step in plan["steps"]:
                try:
                    # Execute the step
                    result = await self._execute_step(step)
                    results["steps"].append({
                        "step": step,
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    results["steps"].append({
                        "step": step,
                        "error": str(e),
                        "success": False
                    })
                    results["errors"].append(str(e))
                    results["success"] = False
                    
                    # Attempt rollback if step fails
                    await self._rollback_step(step)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing deployment plan: {str(e)}")
            raise

    async def _execute_step(self, step: Dict) -> Dict:
        """Execute a deployment step."""
        try:
            if step["type"] == "command":
                # Execute shell command
                result = subprocess.run(
                    step["command"],
                    shell=True,
                    capture_output=True,
                    text=True
                )
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            elif step["type"] == "file_operation":
                # Perform file operation
                if step["operation"] == "copy":
                    shutil.copy2(step["source"], step["destination"])
                elif step["operation"] == "move":
                    shutil.move(step["source"], step["destination"])
                elif step["operation"] == "delete":
                    os.remove(step["path"])
                return {"status": "success"}
            else:
                raise ValueError(f"Unknown step type: {step['type']}")
                
        except Exception as e:
            logger.error(f"Error executing step: {str(e)}")
            raise

    async def _rollback_step(self, step: Dict) -> None:
        """Rollback a failed deployment step."""
        try:
            if step["type"] == "command":
                # Execute rollback command
                if "rollback_command" in step:
                    subprocess.run(
                        step["rollback_command"],
                        shell=True,
                        capture_output=True,
                        text=True
                    )
            elif step["type"] == "file_operation":
                # Perform rollback file operation
                if step["operation"] == "copy":
                    os.remove(step["destination"])
                elif step["operation"] == "move":
                    shutil.move(step["destination"], step["source"])
                elif step["operation"] == "delete":
                    # Restore from backup if available
                    if "backup" in step:
                        shutil.copy2(step["backup"], step["path"])
                        
        except Exception as e:
            logger.error(f"Error rolling back step: {str(e)}")
            raise

    async def _verify_deployment(self, results: Dict) -> Dict:
        """Verify the deployment."""
        try:
            # Prepare the prompt
            prompt = f"""Verify the deployment results:
            {json.dumps(results, indent=2)}
            
            Check:
            1. All steps completed successfully
            2. No errors in logs
            3. System is functioning correctly
            4. Tests are passing
            5. Performance is acceptable
            6. Security requirements are met
            
            Format the response as a JSON object with these categories."""
            
            # Generate verification
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in deployment verification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error verifying deployment: {str(e)}")
            raise

    async def generate_deployment_documentation(self, deployment: Dict) -> str:
        """Generate documentation for the deployment."""
        try:
            # Prepare the prompt
            prompt = f"""Generate documentation for the following deployment:
            {json.dumps(deployment, indent=2)}
            
            Include:
            1. Deployment overview
            2. Environment requirements
            3. Deployment steps
            4. Verification procedures
            5. Rollback procedures
            6. Troubleshooting guide"""
            
            # Generate documentation
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating deployment documentation: {str(e)}")
            raise

    async def analyze_deployment_risks(self, plan: Dict) -> Dict:
        """Analyze deployment risks and provide recommendations."""
        try:
            # Prepare the prompt
            prompt = f"""Analyze the risks in the following deployment plan:
            {json.dumps(plan, indent=2)}
            
            Provide analysis on:
            1. Potential failure points
            2. Security risks
            3. Performance impacts
            4. Data integrity risks
            5. System stability risks
            6. Mitigation strategies
            
            Format the response as a JSON object with these categories."""
            
            # Generate analysis
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in deployment risk analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing deployment risks: {str(e)}")
            raise 