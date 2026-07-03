import json

import httpx


def stream_ds4(config, messages, max_tokens=2048):
    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {"Authorization": "Bearer dsv4-local"}
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            text = parse_openai_stream_line(line)
            if text:
                yield text


def parse_openai_stream_line(line):
    if not line:
        return None
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    obj = json.loads(data)

    # OpenAI chat completions style:
    # {"choices":[{"delta":{"content":"..."}}]}
    choices = obj.get("choices")
    if choices:
        delta = choices[0].get("delta") or {}
        return delta.get("content")

    # OpenAI responses style:
    # {"type":"response.output_text.delta","delta":"..."}
    if obj.get("type") == "response.output_text.delta":
        return obj.get("delta")
    return None
