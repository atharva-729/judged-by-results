"""LLM Foundry client (Classifier A): cached, batched, spend-capped.

Every call is cached under runs/llm_cache/ by hash of (model, prompt), so a rerun
costs nothing. Spend per day is tracked in runs/spend.json and calls stop at
config llm.daily_cap_usd.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

import requests

from jbr import config

RUNS = config.ROOT / "runs"
CACHE = RUNS / "llm_cache"
SPEND = RUNS / "spend.json"


class SpendCapReached(RuntimeError):
    pass


def token() -> str:
    tok = os.environ.get("LLMFOUNDRY_TOKEN")
    env = config.ROOT / ".env"
    if not tok and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLMFOUNDRY_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    if not tok:
        raise RuntimeError("LLMFOUNDRY_TOKEN is not set (add it to .env)")
    return tok


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()


def spent_today() -> float:
    if not SPEND.exists():
        return 0.0
    return json.loads(SPEND.read_text(encoding="utf-8")).get(dt.date.today().isoformat(), 0.0)


def add_spend(usd: float) -> None:
    RUNS.mkdir(exist_ok=True)
    data = json.loads(SPEND.read_text(encoding="utf-8")) if SPEND.exists() else {}
    day = dt.date.today().isoformat()
    data[day] = round(data.get(day, 0.0) + usd, 6)
    SPEND.write_text(json.dumps(data, indent=2), encoding="utf-8")


def complete(prompt: str, cfg: dict, post=requests.post) -> str:
    """One chat completion, from cache if seen before."""
    llm = cfg["llm"]
    key = cache_key(llm["model"], prompt)
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    if spent_today() >= llm["daily_cap_usd"]:
        raise SpendCapReached(f"spent ${spent_today():.2f} today; cap is ${llm['daily_cap_usd']:.2f}")

    resp = post(
        f"{llm['base_url'].rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {token()}:{llm['project']}"},
        json={"model": llm["model"], "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    price = llm["price_per_mtok"]
    usd = (usage.get("prompt_tokens", 0) * price["input"] + usage.get("completion_tokens", 0) * price["output"]) / 1e6
    add_spend(usd)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": llm["model"], "content": content, "usage": usage, "usd": usd}), encoding="utf-8")
    return content


def parse_array(text: str) -> list[dict]:
    """The JSON array in a reply, tolerating a ```json fence."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON array in reply")
    return json.loads(match.group(0))


def classify(items: list[dict], prompt_name: str, cfg: dict, post=requests.post) -> dict[str, dict]:
    """Label ``items`` ({id, fan, text}) in batches; returns {id: label}."""
    template = (config.ROOT / "prompts" / f"{prompt_name}.md").read_text(encoding="utf-8")
    out = {}
    size = cfg["llm"]["batch_size"]
    for i in range(0, len(items), size):
        batch = items[i:i + size]
        lines = "\n".join(f'- id={it["id"]} [{it["fan"]}] {json.dumps(it["text"], ensure_ascii=False)}' for it in batch)
        for label in parse_array(complete(template.replace("{{comments}}", lines), cfg, post)):
            out[str(label["id"])] = label
    return out
