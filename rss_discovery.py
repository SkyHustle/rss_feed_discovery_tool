import sys
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Check if a prompt was provided as a command-line argument
if len(sys.argv) < 2:
    print("Error: Please provide a prompt as a command-line argument.")
    print("Usage: python rss_discovery.py 'your prompt here'")
    sys.exit(1)

# Get the prompt from command-line arguments
PROMPT = sys.argv[1]
print(f"Running with prompt: {PROMPT}")

# Get the API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not found in environment variables.")
    print("Please make sure your .env file contains the OPENROUTER_API_KEY and it's loaded correctly.")
    sys.exit(1)

# Verify the API key is loaded (only showing the first few characters for security)
if len(api_key) > 10:
    print(f"API key loaded: {api_key[:5]}...{api_key[-5:]}")
else:
    print("Warning: API key seems too short")

# Initialize the LLM with OpenRouter
try:
    llm = ChatOpenAI(
        model="google/gemma-2-9b-it:free",  # Using the free Gemma 2 9B model
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Test the connection by sending a simple message
    messages = [HumanMessage(content=PROMPT)]
    response = llm.invoke(messages)
    
    print("\nResponse from AI:\n")
    print(response.content)
    
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your OPENROUTER_API_KEY in the .env file and ensure you have an internet connection.")