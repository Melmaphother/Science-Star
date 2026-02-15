#!/usr/bin/env python3
"""
Test HF_TOKEN (Hugging Face read token).

Uses official huggingface_hub HfApi per:
https://huggingface.co/docs/huggingface_hub/main/package_reference/hf_api
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


def test_hf_token() -> bool:
    """
    Test Hugging Face token via HfApi.whoami().

    Returns:
        bool: True if token works, False otherwise.
    """
    token = os.getenv("HF_TOKEN")
    if not token or token.strip() == "":
        print("❌ HF_TOKEN not set in .env")
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user = api.whoami()
        print(f"✅ HF_TOKEN OK: logged in as {user.get('name', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ HF_TOKEN failed: {e}")
        return False


if __name__ == "__main__":
    success = test_hf_token()
    sys.exit(0 if success else 1)
