"""LLM processor for natural language interface."""

import json
import logging
import os
from typing import Any, Dict, Generator, Optional

from openai import OpenAI

# Setup logging
# BUG FIX: this module previously created a logs/ directory and attached a
# DEBUG FileHandler at IMPORT time - a side-effectful import that wrote to
# disk on any import (including test collection) and stacked duplicate
# handlers on re-import. Standard module logging only; callers configure
# handlers.
logger = logging.getLogger(__name__)


class LLMProcessor:
    """Processes prompts using LLM and handles streaming responses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        # BUG FIX (verified by execution): the OpenAI client was constructed
        # eagerly here and raises OpenAIError when OPENAI_API_KEY is unset,
        # so LLMProcessor() crashed at CONSTRUCTION for any non-OpenAI
        # setup. app.py's try/except silently set the session component to
        # None - on a Claude-configured install (the platform default) this
        # component had never successfully initialized. The client is now
        # created lazily on first use, and a missing key surfaces as a
        # clear error from process()/moderation rather than a boot crash.
        self._client: Optional[OpenAI] = None

        # Load moderation categories
        self.moderation_categories = {
            "hate": True,
            "hate/threatening": True,
            "harassment": True,
            "harassment/threatening": True,
            "self-harm": True,
            "self-harm/intent": True,
            "self-harm/instructions": True,
            "sexual": True,
            "sexual/minors": True,
            "violence": True,
            "violence/graphic": True,
        }

        logger.info("LLMProcessor initialized with moderation categories")

        # Removed return statement - __init__ should not return values

    @property
    def client(self) -> OpenAI:
        """Lazily-constructed OpenAI client (raises with a clear message
        only when an OpenAI-backed call is actually attempted)."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "LLMProcessor requires OPENAI_API_KEY; the platform's "
                    "active LLM is configured elsewhere (config/llm_config)."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def process(self, prompt: str) -> str:
        """Process a prompt and get response.

        Args:
            prompt: Input prompt string

        Returns:
            Response string
        """
        try:
            # Check prompt for unsafe content
            if self.is_unsafe_content(prompt):
                logger.warning("Unsafe content detected in prompt")
                raise ValueError("Prompt contains unsafe content")

            # Get response from LLM
            response = self.client.chat.completions.create(
                model=self.config.get("model", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1000),
            )

            # Extract and validate response
            content = response.choices[0].message.content

            # Check response for unsafe content
            if self.is_unsafe_content(content):
                logger.warning("Unsafe content detected in response")
                raise ValueError("Response contains unsafe content")

            return content

        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            raise

    def process_stream(self, prompt: str) -> Generator[str, None, None]:
        """Process a prompt and stream the response.

        Args:
            prompt: Input prompt string

        Yields:
            Response chunks
        """
        try:
            # Check prompt for unsafe content
            if self.is_unsafe_content(prompt):
                logger.warning("Unsafe content detected in prompt")
                raise ValueError("Prompt contains unsafe content")

            # Get streaming response from LLM
            stream = self.client.chat.completions.create(
                model=self.config.get("model", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1000),
                stream=True,
            )

            # Process stream
            buffer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    buffer += content

                    # Check buffer for unsafe content
                    if self.is_unsafe_content(buffer):
                        logger.warning("Unsafe content detected in stream")
                        raise ValueError("Stream contains unsafe content")

                    yield content

        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}", exc_info=True)
            raise

    def is_unsafe_content(self, content: str) -> bool:
        """Check if content contains unsafe material.

        Args:
            content: Content to check

        Returns:
            True if content is unsafe, False otherwise
        """
        try:
            # Get moderation results
            response = self.client.moderations.create(input=content)
            results = response.results[0]

            # Check each category
            for category, enabled in self.moderation_categories.items():
                if enabled and getattr(results.categories, category):
                    logger.warning(f"Unsafe content detected in category: {category}")
                    return True

            return False

        except RuntimeError:
            # BUG FIX: a missing/invalid API key previously fell into the
            # generic fail-closed branch below, so every prompt was
            # reported as "contains unsafe content" - a configuration
            # error masquerading as a content violation (verified: "hi"
            # was flagged unsafe with no key set). Configuration errors
            # now propagate with their real message; genuine moderation
            # API failures still fail closed.
            raise
        except Exception as e:
            logger.error(f"Error checking content safety: {str(e)}", exc_info=True)
            return True  # Fail safe

    def validate_json_response(self, response: str) -> Dict[str, Any]:
        """Validate and parse JSON response.

        Args:
            response: Response string

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If response is not valid JSON
        """
        try:
            # Try to parse JSON
            data = json.loads(response)

            # Validate required fields
            required_fields = ["response", "confidence"]
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            # Validate confidence
            if (
                not isinstance(data["confidence"], (int, float))
                or not 0 <= data["confidence"] <= 1
            ):
                raise ValueError("Confidence must be a float between 0 and 1")

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            raise
