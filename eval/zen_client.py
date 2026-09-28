"""Chat client for the opencode zen gateway (OpenAI-compatible).

Kept separate from summarizer/pipeline.py so the shipped code keeps no network credentials and no
vendor-specific behaviour. The key is read from ~/.secrets/opencode_zen_key and never logged.
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List

BASE_URL = os.environ.get("ZEN_BASE_URL", "https://opencode.ai/zen/v1")
# OpenCode Go: same key, separate model list. Muse Spark is served only here, and only through the
# Responses API -- chat/completions and messages both return a bare "Internal server error".
GO_BASE_URL = os.environ.get("ZEN_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
GO_RESPONSES_MODELS = ("muse-spark-",)
# Go models on the Anthropic protocol (per opencode.ai/docs/go endpoints). Written "go:<model>"
# so the same model id can still be reached through zen when that is wanted. Go authenticates with
# x-api-key (a Bearer header is rejected as a missing key) and routes on x-opencode-session.
GO_MESSAGES_PREFIX = "go:"
# OpenRouter: model ids carry a provider prefix ("stealth/union-alpha"). Separate key file.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_PATH = os.path.expanduser("~/.secrets/openrouter_key")
OPENROUTER_PREFIXES = ("stealth/",)
KEY_PATH = os.path.expanduser("~/.secrets/opencode_zen_key")


def load_key() -> str:
    with open(KEY_PATH, encoding="utf-8") as f:
        return f.read().strip()


class ZenChat:
    def __init__(self, model: str, max_tokens: int = 900, temperature: float = 0.0,
                 timeout: int = 600, retries: int = 3, max_tokens_ceiling: int = 32000,
                 no_thinking: bool = False):
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.timeout, self.retries = timeout, retries
        self.max_tokens_ceiling = max_tokens_ceiling
        self.no_thinking = no_thinking   # opt-in escape hatch, never the default
        if self.model.startswith(OPENROUTER_PREFIXES):
            with open(OPENROUTER_KEY_PATH, encoding="utf-8") as f:
                self._key = f.read().strip()
        else:
            self._key = load_key()
        self.session_id = f"eval-{os.getpid()}-{int(time.time())}"
        self.last_error = None

    def _request(self, messages: List[Dict[str, str]]):
        """The gateway speaks three dialects and a model only answers on its own.

        Calling everything as OpenAI chat/completions made the whole Claude, Gemini and GPT range
        look unavailable with HTTP 500 - it was the wrong endpoint, not a missing entitlement.
        """
        if self.model.startswith("claude"):
            system = " ".join(m["content"] for m in messages if m["role"] == "system")
            body = {"model": self.model, "max_tokens": self.max_tokens,
                    "messages": [m for m in messages if m["role"] != "system"]}
            if system:
                body["system"] = system
            return (f"{BASE_URL}/messages", body,
                    {"x-api-key": self._key, "anthropic-version": "2023-06-01"})
        if self.model.startswith("gemini"):
            contents = [{"role": "model" if m["role"] == "assistant" else "user",
                         "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
            body = {"contents": contents,
                    "generationConfig": {"maxOutputTokens": self.max_tokens,
                                         "temperature": self.temperature}}
            system = " ".join(m["content"] for m in messages if m["role"] == "system")
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            return (f"{BASE_URL}/models/{self.model}:generateContent", body,
                    {"x-goog-api-key": self._key})
        if self.model.startswith(GO_MESSAGES_PREFIX):
            system = " ".join(m["content"] for m in messages if m["role"] == "system")
            body = {"model": self.model[len(GO_MESSAGES_PREFIX):], "max_tokens": self.max_tokens,
                    "messages": [m for m in messages if m["role"] != "system"]}
            if system:
                body["system"] = system
            return (f"{GO_BASE_URL}/messages", body,
                    {"x-api-key": self._key, "anthropic-version": "2023-06-01",
                     "x-opencode-session": self.session_id})
        if self.model.startswith(OPENROUTER_PREFIXES):
            body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                    "temperature": self.temperature}
            return (f"{OPENROUTER_BASE_URL}/chat/completions", body,
                    {"Authorization": f"Bearer {self._key}", "X-Title": "meeting-summarizer"})
        if self.model.startswith(GO_RESPONSES_MODELS):
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            body = {"model": self.model, "max_output_tokens": self.max_tokens,
                    "input": [{"role": m["role"], "content": m["content"]}
                              for m in messages if m["role"] != "system"]}
            if system:
                body["instructions"] = system
            return f"{GO_BASE_URL}/responses", body, {"Authorization": f"Bearer {self._key}"}
        if self.model.startswith("gpt"):
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            return (f"{BASE_URL}/responses",
                    {"model": self.model, "input": "\n\n".join(parts),
                     "max_output_tokens": self.max_tokens},
                    {"Authorization": f"Bearer {self._key}"})
        # Thinking stays ON. Empty replies were a budget problem, not a runaway loop: the same
        # prompt used 2.3k-7k reasoning tokens across five runs, so a 4k cap clipped 3 of 5 while
        # 24k answered 5 of 5. Disabling thinking would have traded quality for a symptom.
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "temperature": self.temperature}
        if self.no_thinking:
            body["thinking"] = {"type": "disabled"}
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return f"{BASE_URL}/chat/completions", body, {"Authorization": f"Bearer {self._key}"}

    @staticmethod
    def _truncated(data: dict) -> bool:
        """True when the model hit its output cap, in any of the three dialects."""
        for c in data.get("choices") or []:
            if c.get("finish_reason") == "length":
                return True
        if data.get("stop_reason") == "max_tokens":
            return True
        for c in data.get("candidates") or []:
            if c.get("finishReason") in ("MAX_TOKENS", "LENGTH"):
                return True
        return data.get("status") == "incomplete"

    @staticmethod
    def _extract(data: dict) -> str:
        if "choices" in data:                                   # OpenAI chat
            return (data["choices"][0].get("message", {}) or {}).get("content") or ""
        if "content" in data and isinstance(data["content"], list):     # Anthropic
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        if "candidates" in data:                                # Google
            parts = (data["candidates"][0].get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts)
        if "output" in data:                                    # OpenAI responses
            texts = []
            for item in data["output"]:
                for c in item.get("content", []) or []:
                    if c.get("type") in ("output_text", "text"):
                        texts.append(c.get("text", ""))
            return "".join(texts)
        return ""

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        url, payload, auth = self._request(messages)
        body = json.dumps(payload).encode()
        for attempt in range(self.retries + 2):
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         # Cloudflare rejects urllib's default signature with 403 code 1010
                         # before the request ever reaches the API.
                         "User-Agent": "meeting-summarizer/1.0 (+research eval)",
                         "Accept": "application/json",
                         # The gateway rejects a request with no session id (MissingSessionID).
                         "X-Session-ID": self.session_id, **auth})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                text = self._extract(data)
                if text:
                    return text
                # Reasoning length varies run to run, so a cap that usually suffices will still
                # clip some calls and return nothing. Grow the budget rather than lose the session.
                if self._truncated(data) and self.max_tokens < self.max_tokens_ceiling:
                    self.max_tokens = min(self.max_tokens * 2, self.max_tokens_ceiling)
                    self.last_error = f"empty reply, retrying with max_tokens={self.max_tokens}"
                    body = json.dumps(self._request(messages)[1]).encode()
                    continue
                reasoning = ((data.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content")
                if reasoning:
                    self.last_error = (f"model returned only reasoning ({len(reasoning)} chars) "
                                       f"and no answer")
                else:
                    self.last_error = f"no text in reply: {str(data)[:200]}"
                return text
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode("utf-8", "replace")
                self.last_error = f"HTTP {e.code}: {detail}"
                # 429 and 5xx are worth retrying; a 400 means this prompt will never be accepted.
                if e.code < 500 and e.code != 429:
                    return ""
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                self.last_error = f"{type(e).__name__}: {e}"
            if attempt < self.retries - 1:
                time.sleep(5 * (attempt + 1))
        return ""
