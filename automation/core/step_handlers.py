"""
Step Execution Handlers

This module implements handlers for different types of workflow steps.
Each handler is responsible for executing a specific type of action.
"""

import asyncio
import logging
import subprocess
import aiohttp
import json
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import os

class StepHandler(ABC):
    """Base class for step handlers."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step with the given parameters."""
        pass

    @abstractmethod
    async def validate(self, parameters: Dict[str, Any]) -> bool:
        """Validate the step parameters."""
        pass

class CommandStepHandler(StepHandler):
    """Handler for command execution steps."""
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command."""
        command = parameters.get("command")
        if not command:
            raise ValueError("Command parameter is required")

        # Execute the command
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    command,
                    stdout,
                    stderr
                )
            
            return {
                "status": "success",
                "return_code": process.returncode,
                "stdout": stdout.decode().strip(),
                "stderr": stderr.decode().strip()
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {str(e)}")
            return {
                "status": "error",
                "return_code": e.returncode,
                "stdout": e.stdout.decode().strip() if e.stdout else "",
                "stderr": e.stderr.decode().strip() if e.stderr else "",
                "error": str(e)
            }
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def validate(self, parameters: Dict[str, Any]) -> bool:
        """Validate command parameters."""
        if "command" not in parameters:
            return False
        
        command = parameters["command"]
        if not isinstance(command, str) or not command.strip():
            return False
        
        # TODO: Add more validation if needed
        # - Check for dangerous commands
        # - Validate command syntax
        # - Check command permissions
        
        return True

class APIStepHandler(StepHandler):
    """Handler for API call steps."""
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API call."""
        url = parameters.get("url")
        method = parameters.get("method", "GET")
        headers = parameters.get("headers", {})
        data = parameters.get("data")
        timeout = parameters.get("timeout", 30)
        verify_ssl = parameters.get("verify_ssl", True)

        if not url:
            raise ValueError("URL parameter is required")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data if data else None,
                    timeout=timeout,
                    ssl=verify_ssl
                ) as response:
                    response_data = await response.json()
                    
                    return {
                        "status": "success",
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "data": response_data
                    }
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"API call failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        except Exception as e:
            self.logger.error(f"Error making API call: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def validate(self, parameters: Dict[str, Any]) -> bool:
        """Validate API call parameters."""
        if "url" not in parameters:
            return False
        
        url = parameters["url"]
        if not isinstance(url, str) or not url.strip():
            return False
        
        method = parameters.get("method", "GET")
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            return False
        
        # Validate headers if provided
        headers = parameters.get("headers", {})
        if not isinstance(headers, dict):
            return False
        
        # Validate timeout if provided
        timeout = parameters.get("timeout")
        if timeout is not None and not isinstance(timeout, (int, float)):
            return False
        
        return True

class DataProcessingStepHandler(StepHandler):
    """Handler for data processing steps."""
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing."""
        input_data = parameters.get("input_data")
        operation = parameters.get("operation")
        options = parameters.get("options", {})

        if not input_data or not operation:
            raise ValueError("Input data and operation parameters are required")

        try:
            # Convert input data to DataFrame if it's a list or dict
            if isinstance(input_data, (list, dict)):
                df = pd.DataFrame(input_data)
            elif isinstance(input_data, str):
                # Try to parse as JSON
                try:
                    data = json.loads(input_data)
                    df = pd.DataFrame(data)
                except json.JSONDecodeError:
                    # Assume it's a CSV string
                    df = pd.read_csv(pd.StringIO(input_data))
            else:
                raise ValueError("Unsupported input data format")

            # Perform the requested operation
            result = await self._process_data(df, operation, options)
            
            return {
                "status": "success",
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _process_data(self, df: pd.DataFrame, operation: str, options: Dict[str, Any]) -> Any:
        """Process the data according to the specified operation."""
        if operation == "filter":
            return self._filter_data(df, options)
        elif operation == "aggregate":
            return self._aggregate_data(df, options)
        elif operation == "transform":
            return self._transform_data(df, options)
        elif operation == "join":
            return self._join_data(df, options)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _filter_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """Filter the data based on conditions."""
        conditions = options.get("conditions", [])
        if not conditions:
            return df.to_dict(orient="records")
        
        mask = pd.Series(True, index=df.index)
        for condition in conditions:
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if not all([column, operator, value]):
                continue
                
            if operator == "eq":
                mask &= df[column] == value
            elif operator == "ne":
                mask &= df[column] != value
            elif operator == "gt":
                mask &= df[column] > value
            elif operator == "lt":
                mask &= df[column] < value
            elif operator == "ge":
                mask &= df[column] >= value
            elif operator == "le":
                mask &= df[column] <= value
            elif operator == "in":
                mask &= df[column].isin(value)
            elif operator == "not_in":
                mask &= ~df[column].isin(value)
        
        return df[mask].to_dict(orient="records")

    def _aggregate_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate the data based on group by and aggregation functions."""
        group_by = options.get("group_by", [])
        aggregations = options.get("aggregations", {})
        
        if not group_by or not aggregations:
            return df.to_dict(orient="records")
        
        result = df.groupby(group_by).agg(aggregations)
        return result.reset_index().to_dict(orient="records")

    def _transform_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the data based on transformation rules."""
        transformations = options.get("transformations", [])
        if not transformations:
            return df.to_dict(orient="records")
        
        for transform in transformations:
            column = transform.get("column")
            operation = transform.get("operation")
            value = transform.get("value")
            
            if not all([column, operation]):
                continue
                
            if operation == "add":
                df[column] = df[column] + value
            elif operation == "subtract":
                df[column] = df[column] - value
            elif operation == "multiply":
                df[column] = df[column] * value
            elif operation == "divide":
                df[column] = df[column] / value
            elif operation == "fill_na":
                df[column] = df[column].fillna(value)
            elif operation == "replace":
                df[column] = df[column].replace(value)
        
        return df.to_dict(orient="records")

    def _join_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
        """Join the data with another dataset."""
        other_data = options.get("other_data")
        join_type = options.get("join_type", "inner")
        on = options.get("on")
        
        if not other_data or not on:
            return df.to_dict(orient="records")
        
        other_df = pd.DataFrame(other_data)
        result = pd.merge(df, other_df, on=on, how=join_type)
        return result.to_dict(orient="records")

    async def validate(self, parameters: Dict[str, Any]) -> bool:
        """Validate data processing parameters."""
        if "input_data" not in parameters or "operation" not in parameters:
            return False
        
        operation = parameters["operation"]
        if operation not in ["filter", "aggregate", "transform", "join"]:
            return False
        
        # Validate options if provided
        options = parameters.get("options", {})
        if not isinstance(options, dict):
            return False
        
        # Validate operation-specific options
        if operation == "filter":
            conditions = options.get("conditions", [])
            if not isinstance(conditions, list):
                return False
            for condition in conditions:
                if not isinstance(condition, dict):
                    return False
                if not all(k in condition for k in ["column", "operator", "value"]):
                    return False
        
        elif operation == "aggregate":
            if not isinstance(options.get("group_by", []), list):
                return False
            if not isinstance(options.get("aggregations", {}), dict):
                return False
        
        elif operation == "transform":
            transformations = options.get("transformations", [])
            if not isinstance(transformations, list):
                return False
            for transform in transformations:
                if not isinstance(transform, dict):
                    return False
                if not all(k in transform for k in ["column", "operation"]):
                    return False
        
        elif operation == "join":
            if not isinstance(options.get("other_data"), (list, dict)):
                return False
            if not isinstance(options.get("on"), (str, list)):
                return False
        
        return True

class NotificationStepHandler(StepHandler):
    """Handler for notification steps."""
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification."""
        channel = parameters.get("channel")
        message = parameters.get("message")
        recipients = parameters.get("recipients", [])
        subject = parameters.get("subject", "")
        template = parameters.get("template")
        data = parameters.get("data", {})

        if not channel or not message:
            raise ValueError("Channel and message parameters are required")

        try:
            if channel == "email":
                result = await self._send_email(recipients, subject, message, template, data)
            elif channel == "slack":
                result = await self._send_slack(recipients, message, template, data)
            elif channel == "webhook":
                result = await self._send_webhook(recipients, message, template, data)
            else:
                raise ValueError(f"Unsupported notification channel: {channel}")

            return {
                "status": "success",
                "channel": channel,
                "recipients": recipients,
                "result": result
            }

        except Exception as e:
            self.logger.error(f"Notification failed: {str(e)}")
            return {
                "status": "error",
                "channel": channel,
                "error": str(e)
            }

    async def _send_email(self, recipients: List[str], subject: str, message: str,
                         template: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Send an email notification."""
        try:
            # Get email configuration from environment
            sender_email = os.getenv('EMAIL_FROM')
            sender_password = os.getenv('EMAIL_PASSWORD')
            
            if not sender_email or not sender_password:
                self.logger.error("Email configuration not found in environment variables")
                return {"sent": False, "recipients": recipients}
            
            # Configure email settings
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            # Apply template if provided
            if template:
                # TODO: Implement template rendering
                pass

            # Add message body
            msg.attach(MIMEText(message, "plain"))

            # Send email
            with smtplib.SMTP(os.getenv('SMTP_HOST', 'smtp.gmail.com'), 
                             int(os.getenv('SMTP_PORT', '587'))) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            return {"sent": True, "recipients": recipients}

        except Exception as e:
            self.logger.error(f"Email sending failed: {str(e)}")
            raise

    async def _send_slack(self, recipients: List[str], message: str,
                         template: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a Slack notification."""
        # TODO: Get these from configuration
        webhook_url = "https://hooks.slack.com/services/your-webhook-url"

        try:
            # Apply template if provided
            if template:
                # TODO: Implement template rendering
                pass

            # Prepare message
            payload = {
                "text": message,
                "channel": recipients[0] if recipients else "#general"
            }

            # Send message
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Slack API error: {response.status}")

            return {"sent": True, "channel": recipients[0] if recipients else "#general"}

        except Exception as e:
            self.logger.error(f"Slack notification failed: {str(e)}")
            raise

    async def _send_webhook(self, recipients: List[str], message: str,
                           template: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a webhook notification."""
        if not recipients:
            raise ValueError("Webhook URL is required")

        try:
            # Apply template if provided
            if template:
                # TODO: Implement template rendering
                pass

            # Prepare payload
            payload = {
                "message": message,
                "data": data
            }

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(recipients[0], json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Webhook error: {response.status}")

            return {"sent": True, "url": recipients[0]}

        except Exception as e:
            self.logger.error(f"Webhook notification failed: {str(e)}")
            raise

    async def validate(self, parameters: Dict[str, Any]) -> bool:
        """Validate notification parameters."""
        if "channel" not in parameters or "message" not in parameters:
            return False
        
        channel = parameters["channel"]
        if channel not in ["email", "slack", "webhook"]:
            return False
        
        # Validate recipients
        recipients = parameters.get("recipients", [])
        if not isinstance(recipients, list):
            return False
        
        # Validate subject for email
        if channel == "email":
            subject = parameters.get("subject", "")
            if not isinstance(subject, str):
                return False
        
        # Validate template if provided
        template = parameters.get("template")
        if template is not None and not isinstance(template, str):
            return False
        
        # Validate data if provided
        data = parameters.get("data", {})
        if not isinstance(data, dict):
            return False
        
        return True

class StepHandlerFactory:
    """Factory for creating step handlers."""
    _handlers = {
        "command": CommandStepHandler,
        "api": APIStepHandler,
        "data_processing": DataProcessingStepHandler,
        "notification": NotificationStepHandler
    }

    @classmethod
    def get_handler(cls, action_type: str) -> Optional[StepHandler]:
        """Get a handler for the specified action type."""
        handler_class = cls._handlers.get(action_type)
        if handler_class:
            return handler_class()
        return None

    @classmethod
    def register_handler(cls, action_type: str, handler_class: type) -> None:
        """Register a new handler type."""
        cls._handlers[action_type] = handler_class 