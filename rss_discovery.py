import sys
import json
import time
import os
import re
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import feedparser
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool

# Load environment variables from .env file
load_dotenv()

# Function to validate an RSS feed
def validate_rss_feed(url):
    print(f"Validating RSS feed: {url}")
    try:
        # Make a request to the feed URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the feed
        feed = feedparser.parse(response.content)
        
        # Check if it's a valid RSS feed
        if feed.bozo and not feed.entries:
            print(f"  ❌ Invalid RSS feed format: {url}")
            return False, None
        
        # Check if the feed has entries
        if not feed.entries:
            print(f"  ❌ No entries found in the feed: {url}")
            return False, None
            
        # Check build date if available
        build_date = None
        if hasattr(feed.feed, 'lastbuilddate'):
            build_date = feed.feed.lastbuilddate
        elif hasattr(feed.feed, 'updated'):
            build_date = feed.feed.updated
            
        if build_date:
            try:
                # Parse the date - try different formats
                parsed_date = None
                for date_format in [
                    '%a, %d %b %Y %H:%M:%S %z',      # RFC 822 format
                    '%a, %d %b %Y %H:%M:%S %Z',      # RFC 822 with timezone name
                    '%Y-%m-%dT%H:%M:%S%z',           # ISO 8601
                    '%Y-%m-%dT%H:%M:%SZ',            # ISO 8601 UTC
                    '%Y-%m-%dT%H:%M:%S.%f%z',        # ISO 8601 with microseconds
                    '%Y-%m-%d %H:%M:%S',             # Simple format
                ]:
                    try:
                        parsed_date = datetime.strptime(build_date, date_format)
                        break
                    except ValueError:
                        continue
                        
                if parsed_date and datetime.now() - parsed_date > timedelta(hours=72):
                    print(f"  ❌ Feed last build date is older than 72 hours: {url}")
                    return False, None
            except Exception as e:
                print(f"  ⚠️ Error parsing build date: {e}")
                # Continue validation even if we can't parse the date
        
        # Check the newest entry's date
        newest_entry_date = None
        for entry in feed.entries:
            entry_date = None
            if hasattr(entry, 'published'):
                entry_date = entry.published
            elif hasattr(entry, 'updated'):
                entry_date = entry.updated
                
            if entry_date:
                try:
                    # Parse the date - try different formats
                    parsed_date = None
                    for date_format in [
                        '%a, %d %b %Y %H:%M:%S %z',      # RFC 822 format
                        '%a, %d %b %Y %H:%M:%S %Z',      # RFC 822 with timezone name
                        '%Y-%m-%dT%H:%M:%S%z',           # ISO 8601
                        '%Y-%m-%dT%H:%M:%SZ',            # ISO 8601 UTC
                        '%Y-%m-%dT%H:%M:%S.%f%z',        # ISO 8601 with microseconds
                        '%Y-%m-%d %H:%M:%S',             # Simple format
                    ]:
                        try:
                            parsed_date = datetime.strptime(entry_date, date_format)
                            break
                        except ValueError:
                            continue
                            
                    if parsed_date and (newest_entry_date is None or parsed_date > newest_entry_date):
                        newest_entry_date = parsed_date
                except Exception as e:
                    print(f"  ⚠️ Error parsing entry date: {e}")
                    continue
        
        if newest_entry_date and datetime.now() - newest_entry_date > timedelta(hours=72):
            print(f"  ❌ Newest entry is older than 72 hours: {url}")
            return False, None
            
        # Get feed title for reference
        feed_title = feed.feed.title if hasattr(feed.feed, 'title') else "Unknown"
        
        print(f"  ✅ Valid RSS feed: {feed_title}")
        return True, {"url": url, "title": feed_title}
        
    except Exception as e:
        print(f"  ❌ Error validating feed: {str(e)}")
        return False, None

# Tool for the LLM to discover RSS feeds
@tool
def discover_rss_feeds(query: str) -> str:
    """
    Use this function to search the web for RSS feeds related to the query.
    The function should return a list of potential RSS feed URLs.
    """
    try:
        # Create search tool
        search_tool = DuckDuckGoSearchTool()
        
        # Enhance the search query to target RSS feeds
        enhanced_query = f"{query} filetype:xml OR filetype:rss OR inurl:rss OR inurl:feed"
        print(f"Searching for: {enhanced_query}")
        
        # Search for potential feeds
        results = search_tool.run(enhanced_query)
        
        # Look for URLs in the results that are likely RSS feeds
        potential_feeds = []
        for line in results.split('\n'):
            # Extract URLs from the search results
            urls = re.findall(r'https?://[^\s"\'<>]+', line)
            for url in urls:
                # Look for URLs that are likely RSS feeds
                if any(keyword in url.lower() for keyword in ['rss', 'feed', 'xml', 'atom']):
                    potential_feeds.append(url)
        
        # If we don't have enough potential feeds, try a different search approach
        if len(potential_feeds) < 5:
            # Try a second search with different terms
            site_query = f"{query} news site OR newspaper OR media hawaiian"
            print(f"Looking for Hawaiian news sites: {site_query}")
            
            # Add a delay to avoid rate limiting
            time.sleep(2)
            
            site_results = search_tool.run(site_query)
            
            site_urls = []
            for line in site_results.split('\n'):
                urls = re.findall(r'https?://[^\s"\'<>]+', line)
                site_urls.extend(urls)
            
            # For each site URL, try to construct potential RSS feed URLs
            for site_url in site_urls:
                # Clean up the URL to get the base site
                base_url = re.match(r'(https?://[^/]+)', site_url)
                if base_url:
                    base_url = base_url.group(1)
                    # Append common RSS paths and add to potential feeds
                    potential_feeds.extend([
                        f"{base_url}/rss",
                        f"{base_url}/feed",
                        f"{base_url}/rss.xml",
                        f"{base_url}/atom.xml",
                        f"{base_url}/news/feed",
                        f"{base_url}/feed/rss"
                    ])
        
        # Remove duplicates and limit the number of feeds to avoid overwhelming validation
        unique_feeds = list(set(potential_feeds))
        print(f"Found {len(unique_feeds)} potential RSS feeds")
        
        # Return a reasonable number of feeds to validate
        return json.dumps(unique_feeds[:30])
        
    except Exception as e:
        print(f"Error in discover_rss_feeds: {str(e)}")
        # Don't return an empty list, as that will frustrate the model
        return json.dumps([])

# Tool for the LLM to validate a batch of RSS feeds
@tool
def validate_feeds(feed_urls: list) -> dict:
    """
    Validate a list of RSS feed URLs and return those that are valid.
    A valid feed must be properly formatted XML and have been updated within the last 72 hours.
    """
    global all_valid_feeds
    valid_feeds = []
    
    # Handle if feed_urls is a string (e.g., from JSON)
    if isinstance(feed_urls, str):
        try:
            feed_urls = json.loads(feed_urls)
        except json.JSONDecodeError:
            # If it's a single URL string
            feed_urls = [feed_urls]
    
    for url in feed_urls:
        # Skip URLs we've already validated
        if any(feed["url"] == url for feed in all_valid_feeds):
            print(f"Skipping already validated feed: {url}")
            continue
        
        is_valid, feed_info = validate_rss_feed(url)
        if is_valid:
            valid_feeds.append(feed_info)
            # Add to our global list of valid feeds
            all_valid_feeds.append(feed_info)
    
    return {"valid_feeds": valid_feeds, "count": len(valid_feeds), "total_valid": len(all_valid_feeds)}

def main():
    # Check if a prompt was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Error: Please provide a prompt as a command-line argument.")
        print("Usage: python rss_discovery.py 'your prompt here'")
        sys.exit(1)

    # Get the prompt from command-line arguments
    PROMPT = sys.argv[1]
    print(f"Running with prompt: {PROMPT}")
    
    # Generate a filename based on the prompt
    filename = re.sub(r'[^\w\s]', '', PROMPT.lower())[:20].strip().replace(' ', '_')
    output_file = f"{filename}_rss_feeds_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    
    # Get the API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please make sure your .env file contains the OPENROUTER_API_KEY.")
        sys.exit(1)
    
    # Initialize the LLM with OpenRouter
    try:
        llm = ChatOpenAI(
            model="google/gemini-2.0-flash-exp:free",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Define system prompt
        system_prompt = """You are an assistant that helps find valid RSS feeds on the internet.
        Your task is to search for RSS feeds based on the user's query.
        You need to be thorough and creative in your search to find at least 10 valid RSS feeds.
        
        When you find potential RSS feeds, use the validate_feeds function to check if they are valid.
        A valid feed must be properly formatted XML and have been updated within the last 72 hours.
        
        Follow this process:
        1. Use discover_rss_feeds to search for potential RSS feeds with specific search terms
        2. Use validate_feeds to check which feeds are valid
        3. If you don't have enough valid feeds, try again with different search terms
        
        Be strategic in your search:
        - For news sites, try main domain + /rss, /feed, /rss.xml, or /atom.xml
        - Look for specialized local news sources related to the region
        - Try different categories: politics, environment, sports, culture
        - Consider blogs, government sites, and university news related to the query
        
        Remember to use different search strategies for different iterations.
        """
        
        # Define the tools
        tools = [discover_rss_feeds, validate_feeds]
        
        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Storage for valid feeds
        global all_valid_feeds
        all_valid_feeds = []
        
        # First iteration instructions
        user_input = f"""Find me all valid RSS feeds that have anything to do with news in Hawaii.
        I need at least 10 valid and active RSS feeds. 
        Use the discover_rss_feeds function to search for feeds, and then the validate_feeds function to check which ones are valid.
        Be thorough and creative in your search."""
        
        # Run the agent in a loop until we have at least 10 valid feeds
        iteration = 1
        max_iterations = 3
        
        while len(all_valid_feeds) < 10:
            print(f"\n=== Iteration {iteration}: Looking for RSS feeds ({len(all_valid_feeds)}/10 found so far) ===")
            
            try:
                # Run the agent
                result = agent_executor.invoke({"input": user_input, "chat_history": []})
                
                # Extract any valid feeds from the result
                content = result.get("output", "")
                print(f"\nAgent response: {content}")
                
                print(f"\nValid feeds found so far: {len(all_valid_feeds)}/10")
                for i, feed in enumerate(all_valid_feeds, 1):
                    print(f"  {i}. {feed['title']} - {feed['url']}")
                
                # If we've found enough feeds, break the loop
                if len(all_valid_feeds) >= 10:
                    print("Successfully found at least 10 valid RSS feeds!")
                    break
                    
                # Sleep a bit to avoid rate limiting
                print("Waiting a moment before continuing...")
                time.sleep(5)  # Increased delay to help with rate limiting
                
                # Update the user input for the next iteration
                feed_urls = [feed["url"] for feed in all_valid_feeds]
                user_input = f"""I found {len(all_valid_feeds)} valid RSS feeds so far: {feed_urls}
                I need {10 - len(all_valid_feeds)} more valid RSS feeds about news in Hawaii.
                Please search for different feeds than the ones I already have.
                Be more creative in your search terms and try to find local news sources, blogs, or specialized news sites.
                Look for Hawaii-specific topics like local politics, tourism, environment, culture, or sports."""
                
            except Exception as e:
                print(f"Error during iteration {iteration}: {str(e)}")
                # If an iteration fails, try to continue with a different approach
                user_input = "Find me RSS feeds for major Hawaii news sources and newspapers"
                print("Trying a simpler approach for the next iteration")
                time.sleep(3)
            
            iteration += 1
            if iteration > max_iterations:  # Safety limit
                print(f"Reached maximum number of iterations ({max_iterations}). Stopping search.")
                print(f"Found {len(all_valid_feeds)} valid feeds, which is less than the target of 10.")
                # Continue to save what we found
                break
        
        # Save the valid feeds to a JSON file - we always create only one file per run
        with open(output_file, 'w') as f:
            json.dump({"feeds": all_valid_feeds, "count": len(all_valid_feeds), "query": PROMPT}, f, indent=2)
        
        print(f"\n✅ Found {len(all_valid_feeds)} valid RSS feeds. Saved to {output_file}")
        if len(all_valid_feeds) < 10:
            print("Note: Found fewer than 10 valid feeds. You might want to run the script again with a different prompt.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your OPENROUTER_API_KEY and ensure you have an internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()