"""LLM client for interfacing with OpenAI-compatible APIs."""

import asyncio
import logging
import os
from typing import List, Any, Optional
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for ChatOpenAI client with fallback to async invocation."""

    def __init__(self, model: Optional[str] = None, max_tokens: int = 1024) -> None:
        """Initialize LLM client with optional model override."""
        self.model = model or os.getenv("LLM_MODEL")
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self) -> ChatOpenAI:
        """Lazily initialize the ChatOpenAI client."""
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                openai_api_base=os.getenv("LLM_URL"),
                openai_api_key=os.getenv("LLM_API_KEY"),
                temperature=0.1,
                max_tokens=self.max_tokens,
            )
        return self._client

    def invoke(self, messages: List[Any]) -> Any:
        """
        Invoke the LLM with fallback to async if sync client unavailable.

        Args:
            messages: List of LangChain message objects.

        Returns:
            The LLM response object.

        Raises:
            ValueError: If neither sync nor async invocation is possible.
        """
        client = self._get_client()
        try:
            return client.invoke(messages)
        except ValueError as exc:
            if "Sync client is not available" not in str(exc):
                raise
            logger.info("Sync client unavailable, invoking async model")
            if not hasattr(client, "ainvoke"):
                raise
            return asyncio.run(client.ainvoke(messages))
