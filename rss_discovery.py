import sys
import json
import time
import os
import re
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import feedparser
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import concurrent.futures
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Function to normalize URL for comparison
def normalize_url(url):
    # Remove trailing slash
    if url.endswith('/'):
        url = url[:-1]
    # Ensure lowercase for comparison
    return url.lower()

# Function to find potential RSS feed URLs from a base website URL
def find_rss_feeds_from_site(base_url):
    # Prioritize /feed/ as the main path based on examples
    common_paths = [
        "/feed/",  # Primary path to try first
        "/feed",
        "/rss/",
        "/rss",
        "/news/feed/",
        "/index.rss"
    ]
    
    result = []
    # Clean up URL
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Handle URLs that might already contain feed paths
    if any(feed_path in base_url.lower() for feed_path in ['/feed', '/rss']):
        result.append(base_url)
    
    # Try common RSS feed paths
    for path in common_paths:
        result.append(f"{base_url}{path}")
    
    return result

# Function to safely compare dates, handling timezone-aware and naive datetimes
def is_recent_date(date_obj, hours=72):
    # If date is None, we can't determine if it's recent
    if date_obj is None:
        return True
    
    now = datetime.now(timezone.utc)
    
    # If date is naive (no timezone), assume UTC
    if date_obj.tzinfo is None:
        date_obj = date_obj.replace(tzinfo=timezone.utc)
    
    # Compare dates safely
    return (now - date_obj) <= timedelta(hours=hours)

# Function to validate an RSS feed
def validate_rss_feed(url):
    try:
        # Make a request to the feed URL
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        })
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the feed
        feed = feedparser.parse(response.content)
        
        # Check if it's a valid RSS feed
        if feed.bozo and not feed.entries:
            return False, None
        
        # Check if the feed has entries
        if not feed.entries:
            return False, None
            
        # Get feed title for reference
        feed_title = feed.feed.title if hasattr(feed.feed, 'title') else "Unknown"
        
        # Get domain for source classification
        domain = urlparse(url).netloc
        
        # Consider it valid if it has entries
        return True, {"url": url, "title": feed_title, "domain": domain}
        
    except Exception as e:
        return False, None

# Function to ask the LLM for relevant news websites
def get_news_websites_from_llm(prompt, already_checked=None):
    # Get the API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please make sure your .env file contains the OPENROUTER_API_KEY.")
        sys.exit(1)
    
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            model="google/gemini-2.0-flash-exp:free",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create the prompt for the LLM
        if already_checked and len(already_checked) > 0:
            system_message = f"""You are a helpful assistant that finds relevant news websites.
            
            I need news websites for "{prompt}".
            
            I have already checked these websites:
            {', '.join(already_checked[:20])}
            
            Please provide at least 15 NEW news websites that are relevant to "{prompt}".
            Return ONLY the full URLs (including https://) as a bulleted list.
            Don't include any websites from the list I've already checked.
            Focus on regional and local news sources that are most relevant.
            Try to provide a diverse range of different sources.
            """
        else:
            system_message = f"""You are a helpful assistant that finds relevant news websites.
            
            I need news websites for "{prompt}".
            
            Please provide at least 15 news websites that are relevant to "{prompt}".
            Return ONLY the full URLs (including https://) as a bulleted list.
            Focus on regional and local news sources that are most relevant.
            Try to provide a diverse range of different sources.
            """
        
        # Send the message to the LLM
        messages = [HumanMessage(content=system_message)]
        response = llm.invoke(messages)
        
        # Extract website URLs from the response
        websites = []
        for line in response.content.split('\n'):
            # Look for URLs in the line
            if '://' in line:
                # Extract URLs
                urls = re.findall(r'https?://[^\s\)\]\"\'\<\>]+', line)
                websites.extend(urls)
        
        # Clean up URLs
        cleaned_websites = []
        for url in websites:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:]+$', '', url)
            # Ensure URL has a scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            # Normalize the URL to remove trailing slashes for better comparison
            if url.endswith('/'):
                url = url[:-1]
            cleaned_websites.append(url)
        
        return cleaned_websites
    
    except Exception as e:
        print(f"Error getting websites from LLM: {e}")
        return []

# Process a single website to find and validate RSS feeds (for parallel processing)
def process_website(website, valid_feeds_dict):
    result_feeds = []
    
    # Skip if we already have enough feeds
    if len(valid_feeds_dict) >= 10:
        return website, result_feeds
    
    print(f"Checking website: {website}")
    
    # Find potential RSS feed URLs for this website
    potential_feeds = find_rss_feeds_from_site(website)
    
    # Process all feeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(potential_feeds)) as executor:
        future_to_url = {executor.submit(validate_rss_feed, url): url for url in potential_feeds}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                is_valid, feed_info = future.result()
                if is_valid:
                    print(f"✅ Valid feed: {feed_info['title']} - {url}")
                    result_feeds.append(feed_info)
            except Exception as exc:
                print(f"Error processing {url}: {exc}")
    
    return website, result_feeds

# Deduplicate feeds to ensure variety
def deduplicate_feeds(feeds, max_feeds=10):
    if not feeds:
        return []
    
    # First, normalize URLs and group by domain
    domain_grouped = {}
    normalized_urls = {}
    
    for feed in feeds:
        domain = feed.get('domain', urlparse(feed['url']).netloc)
        norm_url = normalize_url(feed['url'])
        
        # Add to domain group
        if domain not in domain_grouped:
            domain_grouped[domain] = []
        
        # Only add if we don't already have this URL (normalized)
        if norm_url not in normalized_urls:
            domain_grouped[domain].append(feed)
            normalized_urls[norm_url] = feed
    
    # Create a deduplicated list ensuring variety
    result = []
    
    # First, take one feed from each domain until we reach max_feeds
    domains = list(domain_grouped.keys())
    while len(result) < max_feeds and domains:
        for domain in list(domains):  # Create a copy to safely modify during iteration
            if not domain_grouped[domain]:
                domains.remove(domain)
                continue
                
            feed = domain_grouped[domain].pop(0)
            result.append(feed)
            
            if len(result) >= max_feeds:
                break
    
    return result[:max_feeds]

def main():
    # Check if a prompt was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Error: Please provide a prompt as a command-line argument.")
        print("Usage: python rss_discovery.py 'your prompt here'")
        sys.exit(1)

    # Get the prompt from command-line arguments
    PROMPT = sys.argv[1]
    print(f"Running with prompt: {PROMPT}")
    
    # Ensure feeds directory exists
    feeds_dir = "feeds"
    os.makedirs(feeds_dir, exist_ok=True)
    
    # Generate a filename based on the prompt
    filename = re.sub(r'[^\w\s]', '', PROMPT.lower())[:20].strip().replace(' ', '_')
    output_file = os.path.join(feeds_dir, f"{filename}_rss_feeds_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    
    # Storage for feeds we've found and websites we've checked
    valid_feeds = []
    checked_websites = []
    max_model_requests = 3
    model_requests = 0
    
    while len(valid_feeds) < 10 and model_requests < max_model_requests:
        model_requests += 1
        print(f"\n=== Model request {model_requests}/{max_model_requests}: Getting news websites from LLM ===")
        
        # Get websites from LLM
        websites = get_news_websites_from_llm(PROMPT, checked_websites)
        
        # Filter out websites we've already checked
        websites = [site for site in websites if site not in checked_websites]
        
        print(f"Found {len(websites)} new websites to check:")
        for i, site in enumerate(websites, 1):
            print(f"  {i}. {site}")
        
        if not websites:
            print("No new websites to check. Moving on...")
            continue
        
        # Process websites in parallel
        print("\n=== Processing websites to find valid RSS feeds ===")
        
        # We'll use a dictionary for atomic updates during parallel processing
        valid_feeds_dict = {normalize_url(feed["url"]): feed for feed in valid_feeds}
        
        all_new_feeds = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(websites))) as executor:
            future_to_site = {executor.submit(process_website, site, valid_feeds_dict): site for site in websites}
            for future in concurrent.futures.as_completed(future_to_site):
                site = future_to_site[future]
                try:
                    site, new_feeds = future.result()
                    # Remember we checked this website
                    checked_websites.append(site)
                    
                    # Collect all new feeds
                    all_new_feeds.extend(new_feeds)
                        
                except Exception as exc:
                    print(f"Error processing {site}: {exc}")
                    checked_websites.append(site)
        
        # Deduplicate all feeds found so far
        deduped_new_feeds = deduplicate_feeds(all_new_feeds)
        
        # Add only new feeds to our collection
        for feed in deduped_new_feeds:
            norm_url = normalize_url(feed["url"])
            if norm_url not in valid_feeds_dict and len(valid_feeds) < 10:
                valid_feeds.append(feed)
                valid_feeds_dict[norm_url] = feed
                print(f"Added feed: {feed['title']} ({len(valid_feeds)}/10)")
        
        # If we have enough feeds after deduplication, break the loop
        if len(valid_feeds) >= 10:
            break
    
    # Final deduplication to ensure variety
    valid_feeds = deduplicate_feeds(valid_feeds, 10)
    
    # Display the final results
    print(f"\n=== Found {len(valid_feeds)}/10 valid RSS feeds ===")
    for i, feed in enumerate(valid_feeds, 1):
        print(f"  {i}. {feed['title']} - {feed['url']}")
    
    # Save the valid feeds to a JSON file
    with open(output_file, 'w') as f:
        json.dump({"feeds": valid_feeds, "count": len(valid_feeds), "query": PROMPT}, f, indent=2)
    
    print(f"\n✅ Found {len(valid_feeds)} valid RSS feeds. Saved to {output_file}")
    if len(valid_feeds) < 10:
        print("Note: Found fewer than 10 valid feeds. You might want to run the script again with a different prompt.")

if __name__ == "__main__":
    main()