"""
Shared LLM client — OpenAI-compatible endpoint (chat.elpai.org).
All scripts import get_client() and call_llm() from here.

Set your API key via environment variable:
    export 
"""

import os
import re
import json
from openai import OpenAI

BASE_URL = "https://chat.elpai.org/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
def get_client() -> OpenAI:
    api_key = os.environ.get("ELPAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ELPAI_API_KEY is not set.\n"
            "Run:  export ELPAI_API_KEY='your_key_here'"
        )
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def call_llm(
    client: OpenAI,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request and return the assistant text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def call_llm_json(
    client: OpenAI,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> dict | list:
    """Call LLM and parse the response as JSON (strips markdown fences if present)."""
    raw = call_llm(client, system, user, model=model, max_tokens=max_tokens)
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)
