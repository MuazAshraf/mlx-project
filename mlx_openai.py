"""
Local mlx_lm chat in ONE file.

Run:  python mlx_openai.py
It starts the mlx_lm server (model path baked in below), waits until it's
ready, then drops you into a streaming chat. Type 'quit' to exit -- the server
is shut down for you.

    pip install mlx-lm openai
"""

import atexit
import subprocess

from openai import OpenAI

MODEL_PATH = "/Users/mojoservo/.omlx/models/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"

# 1. start the server (baked-in model -- nothing to type)
server = subprocess.Popen(["mlx_lm.server", "--model", MODEL_PATH])
atexit.register(server.terminate)   # always clean up the server on exit

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# 2. wait until the model is loaded and the server answers
print("Loading model...", flush=True)
while True:
    try:
        client.models.list()
        break
    except Exception:
        try:
            server.wait(timeout=1)          # sleeps 1s; raises if still running
        except subprocess.TimeoutExpired:
            continue
        raise SystemExit("server exited before it was ready")

# 3. chat
conversation = [{"role": "system", "content": "You are a concise assistant."}]
print("Ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break

    conversation.append({"role": "user", "content": user_input})

    print("Bot: ", end="", flush=True)
    full = ""
    stream = client.chat.completions.create(
        model="local",                     # ignored; server uses its loaded model
        messages=conversation,
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full += delta
    print("\n")

    conversation.append({"role": "assistant", "content": full})
