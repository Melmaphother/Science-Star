#!/usr/bin/env python3
"""
Test LLM API: OPENAI_BASE_URL + OPENAI_API_KEY.

Uses official OpenAI Python client (OpenAI-compatible API):
https://platform.openai.com/docs/api-reference
"""

import os
import sys
from pathlib import Path

# Load .env from project root
_root = Path(__file__).resolve().parent.parent
_dotenv = _root / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)


def test_llm_api() -> bool:
    """
    Test OpenAI-compatible API via openai.OpenAI client.

    Tries models.list() first, then chat.completions if needed.

    Returns:
        bool: True if API works, False otherwise.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or api_key.strip() == "":
        print("❌ OPENAI_API_KEY not set in .env")
        return False

    base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Try list models (lightweight)
        try:
            list(client.models.list())
            print("✅ OPENAI_BASE_URL + OPENAI_API_KEY OK")
            return True
        except Exception:
            pass

        # Fallback: minimal chat completion
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5,
        )
        if resp and resp.choices:
            print("✅ OPENAI_BASE_URL + OPENAI_API_KEY OK")
            return True

        print("❌ LLM API: unexpected response")
        return False
    except Exception as e:
        print(f"❌ LLM API failed: {e}")
        return False


if __name__ == "__main__":
    success = test_llm_api()
    sys.exit(0 if success else 1)
