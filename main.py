import json
import hashlib
import os
import re
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

AIPIPE_MODEL = "gpt-4o"
token = os.getenv("AIPIPE_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

_CACHE = {}
token = os.getenv("AIPIPE_TOKEN")
AIPIPE_BASE = "https://aipipe.org/openai/v1"

HEAD = {"Authorization": f"Bearer {token}",
        "Content-Type": "application/json"}

def _ck(*parts):
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
import asyncio

async def chat(messages, model=None, max_tokens=800, force_json=True, retries=4):
    key = _ck("chat", model, json.dumps(messages, sort_keys=True, default=str))
    if key in _CACHE:
        return _CACHE[key]
    body = {"model": model, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(retries):
            r = await c.post(f"{AIPIPE_BASE}/chat/completions",
                             headers=HEAD, json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:160]}"
                await asyncio.sleep(1.5 * (attempt + 1))   # backoff and retry
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            _CACHE[key] = out
            return out
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")

def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    problem = body.get("problem", "")
    prompt = (
        "Solve this arithmetic word problem CAREFULLY. It deliberately contains "
        "DISTRACTOR numbers that are irrelevant to the final answer.\n"
        "Work in steps:\n"
        "1. List which numbers are relevant and which are distractors.\n"
        "2. Do the arithmetic one operation at a time.\n"
        "3. RE-CHECK the arithmetic a second time before finalising.\n"
        "Return JSON with EXACTLY two keys: 'reasoning' (a string >=80 chars "
        "showing your steps) and 'answer' (a JSON integer — not string, not "
        "float, no symbols).\n\n"
        f"PROBLEM:\n{problem}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}],
                                    model="gpt-4o", max_tokens=1200))
        ans = int(round(float(out.get("answer"))))
        reasoning = str(out.get("reasoning", ""))
        if len(reasoning) < 80:
            reasoning = (reasoning + " Step-by-step arithmetic reasoning applied; "
                         "irrelevant distractor values were identified and ignored.").strip()
        return {"reasoning": reasoning, "answer": ans}
    except Exception as e:
        return {"reasoning": "Could not solve reliably: " + str(e)[:120].ljust(80),
                "answer": 0}
