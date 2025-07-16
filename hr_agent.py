from fastapi import FastAPI, Request
import openai
import chromadb
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("⚠️ OPENAI_API_KEY is missing! Add it to your .env file.")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./hr_vector_db")
collection = chroma_client.get_collection(name="hr_docs")

# Initialize FastAPI
app = FastAPI()

def search_knowledge_base(query):
    """Finds the most relevant HR policy content."""
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(input=[query], model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding  # Corrected new API format

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Retrieve top 3 relevant chunks
    )

    retrieved_docs = [doc for doc in results["documents"][0]]
    metadata = [meta for meta in results["metadatas"][0]]

    return retrieved_docs, metadata

@app.post("/chat")
async def chat(request: Request):
    """HR Policy Agent - Fetches relevant knowledge & refines it via GPT-4o."""
    data = await request.json()
    user_message = data.get("message")

    # Retrieve relevant knowledge
    relevant_docs, metadata = search_knowledge_base(user_message)

    # Format retrieved text for better response
    context = "\n\n".join([f"{meta['heading']} - {meta['subheading']}:\n{doc}" for doc, meta in zip(relevant_docs, metadata)])

    # Generate refined response using GPT-4o
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an HR assistant. Answer HR-related queries based on company policy."},
            {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {user_message}"}
        ]
    )

    return {
        "response": response.choices[0].message.content,
        "references": [meta['heading'] for meta in metadata]  # Return document sources
    }
