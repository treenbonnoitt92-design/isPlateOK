import os
from typing import Any, Dict, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"


class JevClient:
    """
    Client for OpenRouter non-autoregressive Decisions API using Jev model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        raw_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        if not raw_key:
            raise RuntimeError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key."
            )
        self.api_key = raw_key.strip('"\'')
        
        raw_model = model if model is not None else (os.getenv("OPENROUTER_API_MODEL") or DEFAULT_MODEL)
        self.model = raw_model.strip('"\'')
        self.endpoint = base_url or DEFAULT_ENDPOINT

    def _build_payload(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": self.model,
            "state": state,
            "questions": questions,
        }

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def predict(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Synchronously submit state and questions to OpenRouter Decisions API.
        """
        payload = self._build_payload(state, questions)
        headers = self._build_headers()

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(self.endpoint, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter Decisions API failed with HTTP {resp.status_code}: {resp.text}"
                )
            return resp.json()

    async def apredict(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Asynchronously submit state and questions to OpenRouter Decisions API.
        """
        payload = self._build_payload(state, questions)
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.endpoint, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter Decisions API failed with HTTP {resp.status_code}: {resp.text}"
                )
            return resp.json()
