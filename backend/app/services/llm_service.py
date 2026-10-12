"""Provider-agnostic LLM client supporting OpenAI, Anthropic, Gemini, and Ollama."""
import json
import re
from typing import Any, Dict, List, Optional
import httpx
from backend.app.core.config import get_settings


class LLMServiceError(Exception):
    """Safe, redacted exception for LLM provider failures."""
    pass


def redact_error(message: str) -> str:
    """Strip any accidental key, token, or secret from error strings."""
    settings = get_settings()
    if settings.llm_api_key and len(settings.llm_api_key) > 5:
        message = message.replace(settings.llm_api_key, "[REDACTED_API_KEY]")
    # Redact common token patterns
    message = re.sub(r'sk-[A-Za-z0-9_-]{20,}', '[REDACTED_TOKEN]', message)
    message = re.sub(r'Bearer\s+[A-Za-z0-9._-]+', 'Bearer [REDACTED]', message)
    return message


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_configured(self) -> bool:
        return bool(self.settings.llm_api_key and self.settings.llm_api_key.strip()) or self.settings.llm_provider == "ollama"

    async def generate_hypotheses(
        self,
        dataset_profile: Dict[str, Any],
        target_column: str,
        problem_type: str,
        count: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.is_configured():
            raise LLMServiceError("LLM provider is not configured. Please supply LLM_API_KEY in environment.")

        cols = dataset_profile.get("columns", [])
        dtypes = dataset_profile.get("dtypes", {})
        stats = dataset_profile.get("stats", {})

        prompt = (
            f"Dataset Columns: {cols}\n"
            f"Data Types: {json.dumps(dtypes)}\n"
            f"Summary Stats: {json.dumps(stats)}\n"
            f"Target Column: {target_column} ({dtypes.get(target_column, 'unknown')})\n"
            f"Problem Type: {problem_type}\n\n"
            f"Propose {count} engineered feature hypotheses. Rules:\n"
            "- Do not include or use the target column.\n"
            "- Provide concrete transforms (log1p, interaction ratios/products, binning, cyclical, encodings).\n"
            "- Explain why this feature predicts the target mechanism.\n"
            "Respond ONLY with a valid JSON array of objects: "
            '[{"name": "...", "transform_type": "...", "expression": "...", "input_columns": ["col1", "col2"], "hypothesis": "...", "expected_signal": "strong|moderate|weak", "risk": "none|overfit"}]'
        )

        system = (
            "You are an expert ML feature engineering advisor. "
            "Output only JSON. Every proposed feature must be grounded in the provided dataset schema."
        )

        response_text = await self._call_provider(prompt, system)
        return self._parse_json_features(response_text, target_column)

    async def _call_provider(self, prompt: str, system: str) -> str:
        provider = self.settings.llm_provider
        timeout = httpx.Timeout(15.0, connect=10.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if provider == "anthropic":
                    return await self._call_anthropic(client, prompt, system)
                elif provider == "gemini":
                    return await self._call_gemini(client, prompt, system)
                elif provider == "ollama":
                    return await self._call_ollama(client, prompt, system)
                else:
                    # Default: openai-compatible (including LiteLLM / OpenRouter)
                    return await self._call_openai_compatible(client, prompt, system)
        except httpx.HTTPStatusError as e:
            redacted = redact_error(f"HTTP status {e.response.status_code} from provider: {e.response.text[:200]}")
            raise LLMServiceError(redacted) from None
        except httpx.RequestError as e:
            redacted = redact_error(f"Network request error contacting provider: {str(e)}")
            raise LLMServiceError(redacted) from None
        except Exception as e:
            redacted = redact_error(f"Unexpected provider error: {str(e)}")
            raise LLMServiceError(redacted) from None

    async def _call_openai_compatible(self, client: httpx.AsyncClient, prompt: str, system: str) -> str:
        base_url = self.settings.llm_base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        res = await client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]

    async def _call_anthropic(self, client: httpx.AsyncClient, prompt: str, system: str) -> str:
        base_url = self.settings.llm_base_url or "https://api.anthropic.com"
        url = f"{base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": self.settings.llm_api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model or "claude-3-5-sonnet-20241022",
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.3,
        }
        res = await client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        return data["content"][0]["text"]

    async def _call_gemini(self, client: httpx.AsyncClient, prompt: str, system: str) -> str:
        base_url = self.settings.llm_base_url or "https://generativelanguage.googleapis.com"
        model = self.settings.llm_model or "gemini-1.5-flash"
        url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent?key={self.settings.llm_api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
            "generationConfig": {"temperature": 0.3},
        }
        res = await client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_ollama(self, client: httpx.AsyncClient, prompt: str, system: str) -> str:
        base_url = self.settings.llm_base_url or "http://127.0.0.1:11434"
        url = f"{base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.llm_model or "llama3",
            "prompt": f"{system}\n\n{prompt}",
            "stream": False,
        }
        res = await client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "")

    def _parse_json_features(self, raw_text: str, target_column: str) -> List[Dict[str, Any]]:
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            parsed = json.loads(raw_text[start : end + 1])
            clean: List[Dict[str, Any]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                # Sanitize name
                name = re.sub(r'[^a-zA-Z0-9_]', '_', item.get("name", "advisory_feat")).lower()
                inputs = item.get("input_columns", [])
                if target_column in inputs:
                    continue  # Refuse target leakage in advisory hypotheses
                clean.append({
                    "name": name,
                    "transform_type": item.get("transform_type", "custom"),
                    "expression": item.get("expression", f"df['{inputs[0]}']" if inputs else "0"),
                    "input_columns": inputs,
                    "hypothesis": str(item.get("hypothesis", "Advisory feature hypothesis from LLM reasoning.")),
                    "expected_signal": item.get("expected_signal", "moderate"),
                    "risk": item.get("risk", "none"),
                    "source": "llm_advisory",
                })
            return clean
        except json.JSONDecodeError:
            return []
