"""
AXON LLM Client
===============
Model-agnostic client for Google Gemini with a prioritized fallback
chain. If the primary model is unavailable (503) or deprecated (404),
the client automatically moves to the next model in the list.

This is intentional design: a single model outage should not break
inspection. Documented and evaluated as AXON's failure-handling
evidence.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()


class LLMClient:
    """Gemini client with ordered model fallback."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")

        models_env = os.getenv("AXON_LLM_MODELS", "")
        self.models = [m.strip() for m in models_env.split(",") if m.strip()]
        if not self.models:
            raise RuntimeError("AXON_LLM_MODELS not set in .env")

        self.client = genai.Client(api_key=api_key)

    def decide(self, system_prompt: str, user_prompt: str) -> tuple[str, str, int]:
        """
        Send prompts to Gemini. Tries each model in order.
        Returns (response_text, model_used, attempts).
        """
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        attempts = 0
        last_error: Exception | None = None

        for model in self.models:
            attempts += 1
            try:
                r = self.client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                )
                text = (r.text or "").strip()
                if text:
                    return text, model, attempts
            except (errors.ServerError, errors.ClientError) as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"All {len(self.models)} models failed. Last error: {last_error}"
        )

    def decide_json(self, system_prompt: str, user_prompt: str) -> tuple[dict, str]:
        """Same as decide() but parses response as JSON. Returns (parsed, model_used)."""
        text, model, _attempts = self.decide(system_prompt, user_prompt)

        # Strip markdown code fences if present
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

        try:
            return json.loads(text), model
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model returned non-JSON: {text[:300]}") from e