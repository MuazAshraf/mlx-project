from pathlib import Path

import chromadb
from mlx_lm import load, stream_generate
from sentence_transformers import SentenceTransformer


MODEL_PATH = Path(
    "/Users/mojoservo/.omlx/models/"
    "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
)

# LLM
model, tokenizer = load(MODEL_PATH)

# Local embeddings model
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Local persistent vector database
chroma = chromadb.PersistentClient(path="./vector_database")
collection = chroma.get_or_create_collection("client_documents")


def add_documents(documents: list[str]) -> None:
    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[f"document-{i}" for i in range(len(documents))],
        documents=documents,
        embeddings=embeddings,
    )


def retrieve_context(question: str, top_k: int = 3) -> str:
    query_embedding = embedder.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results["documents"][0]
    return "\n\n".join(documents)


# Example documents — baad mein PDFs ke chunks yahan add honge
if collection.count() == 0:
    add_documents([
        "Mojo provides private AI processing on Mac Studio.",
        "Client documents remain stored locally and are not sent to cloud APIs.",
        "MLX runs machine-learning models efficiently on Apple Silicon.",
    ])


conversation_history = []

print("RAG Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"quit", "exit"}:
        break

    context = retrieve_context(user_input)

    rag_message = f"""
Answer using the provided context.

Context:
{context}

Question:
{user_input}

If the answer is not available in the context, say that clearly.
"""

    conversation_history.append({
        "role": "user",
        "content": rag_message
    })

    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    print("Bot: ", end="", flush=True)

    full_response = ""

    for chunk in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024
    ):
        print(chunk.text, end="", flush=True)
        full_response += chunk.text

    print("\n")

    if "</think>" in full_response:
        full_response = full_response.split("</think>")[-1].strip()

    conversation_history.append({
        "role": "assistant",
        "content": full_response
    })