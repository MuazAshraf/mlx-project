"""
Local MLX chat — single file, one model, browser UI.

Loads the model in-process with mlx_lm (no separate server, no extra ports) and
serves a streaming chat page. The model is a reasoning model: its <think> block
is streamed into a collapsible "thinking" area and only the answer is kept in
the conversation history.

Run:
    .venv/bin/python app.py
Then open http://127.0.0.1:7770
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from mlx_lm import load, stream_generate

from chat_store import ChatStore
from ds4_client import stream_ds4
from model_runtime import ModelRuntime
from models import MODELS, get_selected_model
from tools import TOOL_SCHEMAS, run_tool, extract_tool_calls, TOOL_CALL_START

SELECTED_MODEL = get_selected_model()
UI_PORT = 7770
STATIC = Path(__file__).parent / "static"
DATA = Path(__file__).parent / "data"
THINK_END = "</think>"   # the chat template pre-opens <think>; generation closes it
MAX_TOOL_ROUNDS = 5      # cap the generate→call-tool→generate loop

# Tool-use policy. Kept short and unambiguous so the model routes cleanly.
SYSTEM_PROMPT = (
    "You are a helpful assistant with tools. Choose the right tool yourself:\n"
    "- calculate: arithmetic.\n"
    "- web_search: current or uncertain facts. web_fetch: read a specific URL.\n"
    "- list_files, read_file, search_files: anything about this project's files.\n"
    "- get_time: the current date or time.\n"
    "Use a tool when it gives a better answer than guessing. "
    "Don't use tools for simple conversation. Keep answers concise."
)

runtime = ModelRuntime(MODELS, SELECTED_MODEL.key, load)
runtime.load_current()
chat_store = ChatStore(DATA / "chats.db")

app = FastAPI(title="MLX Chat")


@app.get("/api/info")
def info():
    return runtime.list_models()


@app.get("/api/models")
def models():
    return runtime.list_models()


@app.post("/api/model")
async def switch_model(req: Request):
    body = await req.json()
    try:
        selected = runtime.switch(body.get("model"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"current": selected.key, "model": selected.name}


@app.get("/api/chats")
def list_chats():
    return {"chats": chat_store.list_chats()}


@app.post("/api/chats")
async def create_chat(req: Request):
    body = await req.json()
    model_key = body.get("model") or runtime.list_models()["current"]
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'")
    return chat_store.create_chat(model_key)


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    try:
        return chat_store.get_chat(chat_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Chat not found") from e


@app.patch("/api/chats/{chat_id}")
async def update_chat(chat_id: str, req: Request):
    body = await req.json()
    model_key = body.get("model_key")
    if model_key is not None and model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'")
    try:
        return chat_store.update_chat(chat_id, model_key=model_key, title=body.get("title"))
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Chat not found") from e


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    chat_store.delete_chat(chat_id)
    return {"deleted": True}


@app.post("/api/chats/{chat_id}/messages")
async def add_chat_message(chat_id: str, req: Request):
    body = await req.json()
    role = body.get("role")
    if role not in {"user", "assistant", "tool"}:
        raise HTTPException(status_code=400, detail="Invalid message role")
    try:
        return chat_store.add_message(chat_id, role, body.get("content", ""))
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Chat not found") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _run_one_turn(convo, max_tokens):
    """Generate one assistant turn. Streams SSE events for reasoning and (if it's
    a plain answer) the answer text. Returns (post_think_text, tool_calls).
    Post-think content is buffered (not streamed) until we know whether it's a
    tool call or a normal answer, so raw <tool_call> JSON never reaches the UI."""
    model, tokenizer, _ = runtime.get()
    prompt = tokenizer.apply_chat_template(
        convo, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True
    )
    full = ""
    sent_r = sent_a = 0
    mode = None   # None (undecided) | "answer" | "tool"
    for ch in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        full += ch.text
        if THINK_END in full:
            idx = full.index(THINK_END)
            reasoning, post = full[:idx], full[idx + len(THINK_END):]
        else:
            reasoning, post = full, ""
        if len(reasoning) > sent_r:
            yield ("reasoning", reasoning[sent_r:])
            sent_r = len(reasoning)
        if post:
            stripped = post.lstrip()
            if mode is None and stripped:
                if stripped.startswith(TOOL_CALL_START):
                    mode = "tool"
                elif not TOOL_CALL_START.startswith(stripped):
                    mode = "answer"   # not a tool call and not a prefix of one
            if mode == "answer":
                answer = post.lstrip("\n")
                if len(answer) > sent_a:
                    yield ("delta", answer[sent_a:])
                    sent_a = len(answer)
    post = full[full.index(THINK_END) + len(THINK_END):] if THINK_END in full else full
    yield ("__return__", (post, extract_tool_calls(post)))


def _run_ds4_turn(convo, max_tokens):
    cfg = runtime.current()
    full = ""
    for text in stream_ds4(cfg, convo, max_tokens=max_tokens):
        full += text
        yield ("delta", text)
    yield ("__return__", full)


@app.post("/api/chat")
async def chat(req: Request):
    """Stream a reply. Body: {messages: [{role, content}], max_tokens?}.
    Handles a generate→tool-call→generate loop until the model gives an answer."""
    body = await req.json()
    convo = list(body.get("messages", []))
    # Prepend the tool-use policy unless the caller already set a system message.
    if not convo or convo[0].get("role") != "system":
        convo = [{"role": "system", "content": SYSTEM_PROMPT}] + convo
    max_tokens = body.get("max_tokens", 2048)

    def gen():
        try:
            cfg = runtime.current()
            if cfg.backend == "openai":
                final = ""
                for kind, payload in _run_ds4_turn(convo, max_tokens):
                    if kind == "__return__":
                        final = payload
                    else:
                        yield _sse({kind: payload})
                yield _sse({"done": True, "final": final})
                return

            for _ in range(MAX_TOOL_ROUNDS):
                post, calls = "", []
                for kind, payload in _run_one_turn(convo, max_tokens):
                    if kind == "__return__":
                        post, calls = payload
                    else:
                        yield _sse({kind: payload})
                if not calls:
                    yield _sse({"done": True, "final": post.lstrip("\n")})
                    return
                # Record the assistant's tool-call turn, run each tool, then loop.
                convo.append({"role": "assistant", "content": post.strip()})
                for call in calls:
                    yield _sse({"tool_call": call})
                    result = run_tool(call["name"], call["arguments"])
                    yield _sse({"tool_result": {"name": call["name"], "result": result}})
                    convo.append({"role": "tool", "name": call["name"], "content": result})
            yield _sse({"error": f"Stopped after {MAX_TOOL_ROUNDS} tool rounds."})
        except Exception as e:
            yield _sse({"error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn
    print(f"\n  MLX Chat  ->  http://127.0.0.1:{UI_PORT}\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=UI_PORT, log_level="warning")
