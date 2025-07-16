import os
import openai
import chromadb
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("⚠️ OPENAI_API_KEY is missing! Add it to your .env file.")

# Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./hr_vector_db")
collection = chroma_client.get_or_create_collection(name="hr_docs")

# Load HR policy document
file_path = "data/Employee-Handbook.pdf"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

loader = PyPDFLoader(file_path)
documents = loader.load()

# Regex patterns for headings & subheadings
heading_pattern = re.compile(r"^## (.+)", re.MULTILINE)
subheading_pattern = re.compile(r"^### (.+)", re.MULTILINE)

# Default values for headings & subheadings
current_heading = "General Information"
current_subheading = "Miscellaneous"
chunks = []

# Process document line by line
for doc in documents:
    for line in doc.page_content.split("\n"):
        if heading_pattern.match(line):  # Main heading found
            current_heading = line.strip()
            current_subheading = "Miscellaneous"  # Reset subheading
        elif subheading_pattern.match(line):  # Subheading found
            current_subheading = line.strip()
        elif line.strip():  # Normal text
            chunk = {
                "text": f"{current_heading}\n{current_subheading}\n{line.strip()}",
                "heading": current_heading if current_heading else "Unknown",
                "subheading": current_subheading if current_subheading else "Unknown"
            }
            chunks.append(chunk)

# Store document chunks in ChromaDB
client = openai.OpenAI(api_key=openai_api_key)  # Initialize OpenAI client

for i, chunk in enumerate(chunks):
    response = client.embeddings.create(input=[chunk["text"]], model="text-embedding-ada-002")
    embedding = response.data[0].embedding  # Corrected new API format

    # Ensure metadata values are not None
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        metadatas={
            "heading": chunk["heading"],  
            "subheading": chunk["subheading"]
        },
        documents=[chunk["text"]]
    )

print("✅ HR policies stored in ChromaDB successfully!")
