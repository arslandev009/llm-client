"""Smoke test with real APIs. Usage:
    python packages/llm-client/scripts/smoke.py            # uses LLM_PROVIDER
    python packages/llm-client/scripts/smoke.py gemini groq local
"""
import os
import sys

from pydantic import BaseModel

from llm_client import LLM


class City(BaseModel):
    city: str
    country: str


def run(provider: str) -> None:
    print(f"\n=== {provider} ===")
    llm = LLM(provider)
    r = llm.chat([{"role": "user", "content": "Say hello in one short sentence."}], max_tokens=500)
    print("chat :", r.text.strip(), f"| {r.prompt_tokens}+{r.completion_tokens} tok | {r.latency_s:.2f}s")
    try:
        c = llm.chat_json([{"role": "user", "content": "Capital of Pakistan?"}], City, max_tokens=500)
        print("json :", c)
    except Exception as exc:  # noqa: BLE001
        print("json : FAILED ->", repr(exc)[:200])


if __name__ == "__main__":
    for p in sys.argv[1:] or [os.getenv("LLM_PROVIDER", "gemini")]:
        try:
            run(p)
        except Exception as exc:  # noqa: BLE001
            print(f"{p}: ERROR -> {exc}")
