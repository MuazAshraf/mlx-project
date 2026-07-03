from pathlib import Path
from mlx_lm import load, generate

MODEL_PATH = Path("/Users/mojoservo/.omlx/models/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")

model, tokenizer = load(MODEL_PATH)

conversation_history = []

print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit"]:
        break

    conversation_history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False)

    # strip think tags if present
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    conversation_history.append({"role": "assistant", "content": response})

    print(f"\nBot: {response}\n")