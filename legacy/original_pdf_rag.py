from pypdf import PdfReader
import ollama
import numpy as np
import sys


# ==================================================
# 1. Load PDF
# ==================================================

def load_pdf(path):

    reader = PdfReader(path)

    pages = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            pages.append({
                "page": page_num + 1,
                "text": text
            })

    return pages


pages = load_pdf(
    sys.argv[1]
)


# ==================================================
# 2. Chunk text
# ==================================================

def chunk_text(text, chunk_size=1200, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==================================================
# 3. Pages -> chunks
# ==================================================

def create_chunks(pages):

    all_chunks = []

    for page in pages:

        chunks = chunk_text(
            page["text"]
        )

        for chunk_id, chunk in enumerate(chunks):

            all_chunks.append({
                "page": page["page"],
                "chunk_id": chunk_id,
                "text": chunk
            })

    return all_chunks


chunks = create_chunks(pages)

print("Pages:", len(pages))
print("Chunks:", len(chunks))


# ==================================================
# 4. Chunk embeddings
# ==================================================

def embed_chunks(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    response = ollama.embed(
        model="nomic-embed-text",
        input=texts
    )

    for chunk, embedding in zip(
        chunks,
        response["embeddings"]
    ):

        chunk["embedding"] = embedding

    return chunks


chunks = embed_chunks(chunks)

print(
    "Embedding dimension:",
    len(chunks[0]["embedding"])
)


# ==================================================
# 5. Query embedding
# ==================================================

def embed_query(query):

    response = ollama.embed(
        model="nomic-embed-text",
        input=query
    )

    return response["embeddings"][0]


# ==================================================
# 6. Cosine similarity
# ==================================================

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )


# ==================================================
# 7. Retrieve
# ==================================================

def retrieve(query, chunks, top_k=5):

    # 把问题变成向量
    query_embedding = embed_query(query)

    scored_chunks = []

    # 和每一个 chunk 比较
    for chunk in chunks:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        scored_chunks.append({
            **chunk,
            "score": score
        })

    # 相似度由高到低
    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # 返回最相关的 Top-K
    return scored_chunks[:top_k]


# ==================================================
# 8. Test
# ==================================================

query = "What data does this paper use?"

results = retrieve(
    query,
    chunks,
    top_k=3
)


for result in results:

    print("=" * 50)

    print(
        "Page:",
        result["page"]
    )

    print(
        "Score:",
        result["score"]
    )

    print(
        result["text"][:500]
    )

def build_context(results):
    context_parts = []

    for result in results:

        context_parts.append(
             f"[Page{result['page']}]\n"
             f"{result['text']}"
        )
    return "\n\n".join(context_parts)
context = build_context(results)

print(context)


def generate_answer(query,results):
    context =build_context(results)

    prompt = f"""You are a question-answering assistant.
             Answer the question using ONLY the provided context.

             If the answer cannot be found in the context,
             say "I cannot find this information in the document."

             When possible, include the relevant page number.


             Context:
             {context}

             Question:
             {query}

             Answer:
             """
    response = ollama.chat(
      model = "qwen3:4b",
      messages=[
            {
               "role":"user",
               "content":prompt
            }
      ]
    )
    return response["message"]["content"]
answer = generate_answer(
         query,
         results
)

print("\n" + "=" *50)
print("FINAL ANSWER")
print("="*50)
print(answer)
