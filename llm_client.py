"""Legacy LLM client adapter forwarding to backend LLM service."""
import asyncio
import os
from typing import Any, Dict, List, Optional
from backend.app.services.llm_service import LLMService, LLMServiceError


class LLM:
    def __init__(self) -> None:
        self.service = LLMService()

    def generate(self, prompt: str, system: str = "You are a helpful AI assistant.") -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.service._call_provider(prompt, system))
        except LLMServiceError as err:
            return f"LLM Advisory Error: {str(err)}"
