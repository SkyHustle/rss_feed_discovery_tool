import sys
from langchain.llms import OpenRouter
from langchain.agents import initialize_agent, AgentType
import feedparser

# Get prompt from command line
PROMPT = sys.argv[1]

# OpenRouter setup (replace with your API key)
llm = OpenRouter(model="mistral-7b-instruct", api_key="your_openrouter_api_key")

# Define validation tool
def validate_feed(url):
    feed = feedparser.parse(url)
    if feed.entries and (feed.feed.get("updated") or feed.entries[0].get("published")):
        # Placeholder: Add Supabase save later
        print(f"Valid feed found: {url}")
        return True
    return False

# Initialize agent with built-in search and custom tool
agent = initialize_agent(
    tools=[validate_feed],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Run agent with prompt
agent.run(f"Find 10 RSS feeds for: {PROMPT}")