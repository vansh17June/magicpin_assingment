"""Check whether the configured Gemini API key can access the selected model."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from urllib.parse import quote

from bot import AI_TIMEOUT_SECONDS, _ai_settings


def main() -> int:
    try:
        settings = _ai_settings()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return 2
    if settings is None or settings[0] != "gemini":
        print("Gemini is not configured. Set VERA_GEMINI_API_KEY in this PowerShell session.")
        return 2

    _, api_key, model, _ = settings
    model_id = model.removeprefix("models/")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model_id, safe='-._')}"
    )
    request = urllib.request.Request(
        endpoint,
        headers={"x-goog-api-key": api_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=AI_TIMEOUT_SECONDS) as response:
            model_info = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            details = json.loads(exc.read(4096).decode("utf-8")).get("error", {}).get("message", "")
        except (UnicodeDecodeError, json.JSONDecodeError):
            details = ""
        if isinstance(details, str) and api_key in details:
            details = details.replace(api_key, "[redacted]")
        print(f"Gemini model lookup failed: HTTP {exc.code}. {details}".rstrip())
        return 1
    except urllib.error.URLError as exc:
        print(f"Could not reach the Gemini API: {exc.reason}")
        return 1
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Gemini returned an invalid response: {exc}")
        return 1

    methods = model_info.get("supportedGenerationMethods", [])
    print(f"Gemini key can access model: {model_info.get('name', model_id)}")
    if "generateContent" not in methods:
        print(f"Warning: this model does not list generateContent support: {methods}")
        return 1
    print("Model supports generateContent. You can now run: python build_submission.py --use-ai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
