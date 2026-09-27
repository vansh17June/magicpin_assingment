# Vera challenge bot

This submission implements `compose(category, merchant, trigger, customer)` and the HTTP endpoints in the testing brief using only the Python standard library. The composer always selects grounded facts and a call to action with rules. It can optionally use Gemini or an OpenAI-compatible chat-completions model to improve wording without changing the selected call to action.

## Run

```powershell
python bot.py
```

The bot listens on `http://localhost:8080` by default. Set `PORT`, `HOST`, `TEAM_NAME`, `TEAM_MEMBERS`, and `CONTACT_EMAIL` as environment variables when needed. To create the 30 canonical output lines, run:

```powershell
python build_submission.py
```

Run the contract tests with:

```powershell
python -m unittest test_bot -v
```

Run the HTTP service directly with `python bot.py`; it serves the judge API on port 8080 by default. Set `PORT` to change the port.

Build a deployment image with `docker build -t vera-challenge .` and run it with your provider key passed through a secret/environment variable, for example:

```powershell
docker run --rm -p 8080:8080 -e VERA_GEMINI_API_KEY=$env:VERA_GEMINI_API_KEY vera-challenge
```

## Enable AI wording

PowerShell:

```powershell
$env:VERA_GEMINI_API_KEY = "your-new-key"
$env:VERA_GEMINI_MODEL = "gemini-2.5-flash"
python bot.py
```

Create a Gemini API key in Google AI Studio. Since a key was pasted into chat, revoke it and create a replacement before using the integration. Put the replacement in `VERA_GEMINI_API_KEY` locally; do not paste it in chat or commit it. Optional `VERA_GEMINI_MODEL` defaults to `gemini-2.5-flash`. To use another OpenAI-compatible provider instead, set `VERA_OPENAI_API_KEY`, `VERA_OPENAI_MODEL`, and optionally `VERA_OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`); configure only one provider at a time.

With no key, the bot uses rules only. When configured, the API sends composed message drafts—including names, prices, or dates included in the draft—to the selected model provider. Check the provider's free-tier quotas, availability, and terms; free usage is not guaranteed. AI failures return HTTP 503 rather than silently claiming an AI-generated result. At most three AI rewrites are made per tick to stay within the judge's response-time budget.

To generate a model-rewritten JSONL submission explicitly, run `python build_submission.py --use-ai` after configuring the key. This sends each of the 30 drafts to the provider and may consume quota.

## Approach and tradeoffs

The composer selects message patterns by trigger kind, retrieves cited items from the current category digest, and uses numbers and offers only when those facts exist in the pushed contexts. An optional LLM only rewrites the already-grounded draft; CTA, rationale, suppression key, and send-as decision remain rule-driven. The API returns a generic `{{1}}` initial-message template with the composed body as its single parameter; replies are handled as in-session free-form messages. Customer outreach requires recorded opt-in for a matching reminder purpose. The API stores versioned context in memory, deduplicates outbound messages by suppression key, and handles opt-outs, automated replies, clear intent, and requests to wait.

This hybrid approach trades some stylistic freedom for grounding and predictable behavior. It does not provide a semantic guarantee against every unsupported claim or fully translate arbitrary messages into every listed language; review model-generated output and add stronger language-specific testing before production use.
