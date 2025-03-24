# RSS Feed Discovery Tool

A Python script that automates the discovery and validation of RSS feeds based on any topic or prompt. The tool uses an LLM (Large Language Model) to find relevant news websites and then validates their RSS feeds in parallel.

## Features

-   **Topic-Agnostic Discovery**: Find RSS feeds for any topic just by changing the prompt
-   **Fully Autonomous**: Discovers and validates feeds without requiring hardcoded example feeds
-   **Smart Deduplication**: Ensures variety by removing duplicate feeds and prioritizing diverse sources
-   **Parallel Processing**: Validates multiple feeds simultaneously for improved performance
-   **Progressive Discovery**: Makes multiple attempts to find feeds if initial searches don't yield enough results

## Requirements

-   Python 3.8+
-   An OpenRouter API key (for accessing LLMs)

## Installation

1. Clone this repository:

    ```
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project directory with your OpenRouter API key:
    ```
    OPENROUTER_API_KEY=your_api_key_here
    ```

## Usage

Run the script with a prompt describing the type of RSS feeds you want to discover:

```
python rss_discovery.py "Find me all valid RSS feeds that have anything to do with finance technology news"
```

The script will:

1. Ask the LLM for relevant news websites based on your prompt
2. Find and validate RSS feeds from these websites
3. Deduplicate the results to ensure variety
4. Save the valid feeds to a JSON file

The output JSON file will contain up to 10 valid RSS feeds related to your query.

## How It Works

### 1. LLM-Based Website Discovery

The script uses OpenRouter's API to access Google's Gemini model. It sends a prompt asking for news websites related to your query. The LLM returns a list of relevant website URLs.

### 2. RSS Feed Detection

For each website, the script attempts to find RSS feeds by:

-   Checking common RSS feed paths (like `/feed/`, `/rss/`, etc.)
-   Normalizing URLs to avoid duplicates
-   Processing all potential feed URLs in parallel

### 3. Feed Validation

Each potential feed is validated by:

-   Making an HTTP request to the feed URL
-   Parsing the response with feedparser
-   Checking if it's a valid RSS/Atom feed with entries
-   Extracting the feed title and domain information

### 4. Deduplication & Variety

The script ensures variety in the results by:

-   Normalizing URLs to prevent slight variations of the same feed
-   Grouping feeds by domain
-   Taking one feed from each domain before moving to duplicates
-   Limiting the total number of feeds to 10

### 5. Result Storage

Valid feeds are saved to a JSON file named based on your prompt:

```
{
  "feeds": [
    {
      "url": "https://example.com/feed",
      "title": "Example Feed Title",
      "domain": "example.com"
    },
    ...
  ],
  "count": 10,
  "query": "Your original prompt"
}
```

## Configuration

The script has several configurable parameters:

-   `max_model_requests`: Maximum number of times to request website lists from the LLM (default: 3)
-   `common_paths`: List of common RSS feed URL paths to check
-   Feed validation parameters like timeout and user-agent

## Troubleshooting

If the script doesn't find enough valid feeds:

-   Try a more specific prompt
-   Increase the maximum number of model requests
-   Check your internet connection
-   Verify your OpenRouter API key is valid

## Limitations

-   Some websites may block automated requests
-   The LLM might suggest websites that don't have RSS feeds
-   The script is rate-limited by the APIs it uses
-   Some feeds might appear valid but contain outdated content

## License

[Your license information here]

## Acknowledgments

-   OpenRouter for API access to LLMs
-   feedparser library for RSS feed parsing
-   requests library for HTTP requests
