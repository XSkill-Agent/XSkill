"""
Visit tool - visit webpage and extract content
Use Trafilatura to extract webpage content (open source solution)
Fail to fallback to Jina API
"""

import os
import requests
from tools.base import BaseTool
from tools.tool_registry import register_tool

try:
    import trafilatura
except ImportError:
    trafilatura = None  # optional: Jina can be used instead

# Jina API support (fallback when trafilatura fails)
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_AVAILABLE = JINA_API_KEY is not None


@register_tool("visit")
class Visit(BaseTool):
    name = "visit"
    description = "Visit a webpage and extract its main content. Use when you have a specific URL to visit (often after getting a URL from web search or image search). Extracts and returns the main textual content of the webpage."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL of the webpage to visit (must start with http:// or https://)"
            },
            "goal": {
                "type": "string",
                "description": "What information you want to find on this page (helps focus the extraction)"
            }
        },
        "required": ["url", "goal"]
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        if trafilatura is None and not JINA_AVAILABLE:
            raise ImportError("Either trafilatura or JINA_API_KEY is required for Visit tool")
        
        # Configuration
        self.max_content_length = config.get('max_content_length', 5000) if config else 5000
        self.use_llm_summary = config.get('use_llm_summary', True) if config else True  # Default enabled
        self.timeout = config.get('timeout', 15) if config else 15
        
        # API configuration
        self.api_key = config.get('api_key') if config else None
        if not self.api_key:
            print(f"Warning: No API key provided, using environment variable REASONING_API_KEY or EVALUATOR_API_KEY")
        
        self.api_endpoint = config.get('api_endpoint') if config else None
        if not self.api_endpoint:
            print(f"Warning: No API endpoint provided, using environment variable REASONING_END_POINT or EVALUATOR_END_POINT")

        self.model_name = config.get('model_name') if config else None
        if not self.model_name:
            print(f"Warning: No model name provided, using environment variable REASONING_MODEL or EVALUATOR_MODEL")
    
    def call(self, params, **kwargs):
        """
        Visit webpage and extract content
        
        Args:
            params: Dictionary containing url and goal
            **kwargs: Optional llm parameters for content summary
            
        Returns:
            Extracted webpage content or summary
        """
        # Parse parameters
        if isinstance(params, str):
            import json
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return "Error: Invalid parameters format"
        
        url = params.get("url", "")
        goal = params.get("goal", "")
        
        if not url:
            return "Error: No URL provided"
        
        try:
            print(f"[Visit] Fetching URL: {url}")
            print(f"[Visit] Goal: {goal}")
            
            downloaded = None
            # Prioritize using Jina for extraction, then fallback to requests + trafilatura
            content = None
            if JINA_AVAILABLE:
                print("[Visit] Trying Jina API first...")
                downloaded = self._jina_readpage(url)
                if downloaded and not downloaded.startswith("[visit] Failed to read page."):
                    content = downloaded
                    print(f"[Visit] Jina API extraction successful: {len(content)} characters")
            
            # If Jina is not available or fails, fallback to requests + trafilatura
            if not content:
                # Use Trafilatura to extract webpage content
                # Improvement 1: Add custom headers (pretend to be a browser)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                # Use requests with headers to get content
                try:
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    downloaded = response.text
                except Exception as e:
                    print(f"[Visit] Requests failed: {e}, trying trafilatura.fetch_url as fallback")
                    if trafilatura is not None:
                        try:
                            downloaded = trafilatura.fetch_url(url)
                        except Exception as e2:
                            print(f"[Visit] Trafilatura.fetch_url also failed: {e2}")
                
                # Improvement 2: Extract main content, enable favor_recall (improve recall)
                if downloaded and trafilatura is not None:
                    content = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        output_format='markdown',
                        include_links=True,
                        include_tables=True,
                        favor_recall=True,  # Improve recall, reduce missed extraction
                        include_formatting=True,
                        deduplicate=True
                    )
                    
                    if not content:
                        # Try plain text format + favor_recall
                        print("[Visit] Markdown extraction failed, trying plain text with favor_recall")
                        content = trafilatura.extract(
                            downloaded,
                            include_comments=False,
                            output_format='txt',
                            favor_recall=True,
                            deduplicate=True
                        )
                elif downloaded and trafilatura is None:
                    print("[Visit] Trafilatura not available, skipped local extraction")
                
                # If requests+trafilatura still fails, try Jina last (prevent Jina from failing initially)
                if not content and JINA_AVAILABLE:
                    print(f"[Visit] Fallback to Jina API after other methods failed...")
                    content = self._jina_readpage(url)
                    if content and not content.startswith("[visit] Failed to read page."):
                        print(f"[Visit] Jina API extraction successful: {len(content)} characters")
                    else:
                        content = None
            
            # All methods failed
            if not content:
                return f"Error: No content extracted from {url} (Jina, requests, and trafilatura all failed)"
            
            print(f"[Visit] Extracted {len(content)} characters")
            
            # Limit length (avoid too long)
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "\n\n[Content truncated due to length...]"
            
            # If summary function is enabled, use API to summarize content
            if self.use_llm_summary and goal and self.api_key and self.api_endpoint:
                try:
                    summary = self._summarize_with_api(content, goal, url)
                    return summary
                except Exception as e:
                    print(f"[Visit] API summarization failed: {e}, returning raw content")
                    # Return raw content when failed
            
            # Otherwise return extracted content
            return f"Content from {url}:\n\nGoal: {goal}\n\n{content}"
        
        except Exception as e:
            error_msg = f"Error visiting {url}: {str(e)}"
            print(f"[Visit] {error_msg}")
            return error_msg
    
    def _summarize_with_api(self, content, goal, url):
        """
        Use API to summarize webpage content (using WebWatcher's structured prompt)
        
        Args:
            content: Webpage content
            goal: Visiting goal
            url: Webpage URL
            
        Returns:
            Summarized content
        """
        import json
        
        prompt = f"""Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction**: Identify and extract the **most relevant information** from the content. Never miss any important information. Output the **full original context** as far as possible (can be more than three paragraphs)
3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity and judging the contribution of the information to the goal

## **Output Format**
Please respond in JSON format with the following fields:
{{
  "evidence": "Key quotes or facts from the page that are directly relevant to the goal",
  "summary": "A concise summary of how the webpage content answers or relates to the user's goal"
}}"""
        
        try:
            # Build API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes webpage content based on user goals."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 8192
            }
            
            print(f"[Visit] Calling API to summarize content (model: {self.model_name})...")
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                timeout=240
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response content
            summary_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not summary_text:
                raise ValueError("Empty response from API")
            
            try:
                # Extract JSON part (possibly wrapped in markdown code block)
                if '```json' in summary_text:
                    # Extract content from ```json ... ```
                    start = summary_text.find('```json') + 7
                    end = summary_text.find('```', start)
                    json_str = summary_text[start:end].strip()
                elif '```' in summary_text:
                    # Extract content from ``` ... ```
                    start = summary_text.find('```') + 3
                    end = summary_text.find('```', start)
                    json_str = summary_text[start:end].strip()
                elif summary_text.strip().startswith('{'):
                    # Directly JSON
                    json_str = summary_text.strip()
                else:
                    # Try to find first { and last }
                    left = summary_text.find('{')
                    right = summary_text.rfind('}')
                    if left != -1 and right != -1 and left < right:
                        json_str = summary_text[left:right+1]
                    else:
                        # Cannot find JSON, use raw text
                        raise ValueError("No JSON found")
                
                # Parse JSON
                parsed = json.loads(json_str)
                evidence = parsed.get('evidence', '')
                summary = parsed.get('summary', '')
                
                formatted_output = f"The useful information in {url} for user goal '{goal}' as follows:\n\n"
                if evidence:
                    formatted_output += f"**Evidence in page:**\n{evidence}\n\n"
                if summary:
                    formatted_output += f"**Summary:**\n{summary}\n\n"
                
                print(f"[Visit] API summarization successful (evidence: {len(evidence)} chars, summary: {len(summary)} chars)")
                return formatted_output
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # JSON parsing failed, use raw response
                print(f"[Visit] Failed to parse JSON response ({e}), using raw text")
                print(f"[Visit] API summarization successful ({len(summary_text)} chars)")
                return f"Summary from {url}:\n\n{summary_text}"
            
        except Exception as e:
            print(f"[Visit] API call failed: {e}")
            raise  # Re-throw exception, let outer layer catch and return raw content
    
    def _jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina Reader API as fallback.
        
        Args:
            url: The URL to read
            
        Returns:
            str: The webpage content or error message
        """
        if not JINA_AVAILABLE:
            return "[visit] Jina API not available (JINA_API_KEY not set)"
        
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        max_retries = 3
        timeout = 20
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(f"[Visit] Jina API error {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:
                        return "[visit] Failed to read page."
            except Exception as e:
                print(f"[Visit] Jina API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
        
        return "[visit] Failed to read page."

