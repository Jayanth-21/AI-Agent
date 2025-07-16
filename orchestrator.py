from fastapi import FastAPI, Request
import openai
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("⚠️ OPENAI_API_KEY is missing. Please set it in the .env file.")

# Initialize FastAPI
app = FastAPI()

# Define routing for sub-agents
AGENT_ROUTES = {
    "hr_policy": "http://127.0.0.1:8001/chat",
    "pricing": "https://pricing-service-url/chat",
    "complaints": "https://complaint-service-url/chat"
}

# Set a maximum retry limit
MAX_RETRIES = 3

def classify_intent(user_message):
    """Uses OpenAI GPT-4 to classify user intent."""
    client = openai.OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Classify the user's request into one of: 'hr_policy', 'pricing', 'complaints'. If unsure, return 'unknown'."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return "unknown"  # Return 'unknown' instead of crashing

@app.post("/chat")
async def chat(request: Request):
    """Orchestrator receives a message and routes it to the correct sub-agent, with retry handling."""
    data = await request.json()
    user_message = data.get("message")
    
    retry_count = 0  # Track how many times classification has failed

    while retry_count < MAX_RETRIES:
        intent = classify_intent(user_message)

        if intent in AGENT_ROUTES:
            try:
                agent_url = AGENT_ROUTES[intent]
                response = requests.post(agent_url, json={"message": user_message}, timeout=5)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    retry_count += 1  # Retry if the request fails

            except requests.exceptions.RequestException:
                retry_count += 1  # Retry if the request throws an error

        else:
            retry_count += 1  # Retry if classification returns 'unknown'

    # If retries exceed limit, return a final fallback message
    return {
        "response": "I'm having trouble understanding your request. If you would like to speak to a human support agent, please submit the contact form. Have a good day!"
    }
