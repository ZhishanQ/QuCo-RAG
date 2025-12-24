import openai
import time
import logging
from typing import Optional
import re
logger = logging.getLogger(__name__)

class GPTAPIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", max_retries: int = 3):
        """
        Initialize GPT API client
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL
            max_retries: Maximum number of retries
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.max_retries = max_retries
    
    def remove_web_links(self, text):
        """
        Remove web link references from text.

        Supported formats:
        - ([domain.com](url))
        - [text](url)
        - (url)
        """
        # Remove ([domain.com](url?utm_source=openai)) style links
        text = re.sub(r'\s*\(\[[\w\-\.]+\]\(https?://[^\)]+\)\)', '', text)
        
        # Remove [text](url) Markdown links but keep the link text
        text = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', text)
        
        # Remove standalone (url) format
        text = re.sub(r'\s*\(https?://[^\)]+\)', '', text)
        
        return text.strip()

    def chat_completion(self,
                       prompt: str,
                       model: str = "gpt-4o-mini",
                       return_usage: bool = False,
                       use_web_search: bool = False,
                       **kwargs):
        """
        Call GPT responses API with retry mechanism.

        Args:
            prompt: Input prompt text (single string)
            model: Name of the model to use
            return_usage: If True, return tuple of (text, usage_dict), otherwise just text
            use_web_search: If True, enable web search tool for the API call
            **kwargs: Additional parameters (ignored for compatibility)

        Returns:
            str or tuple: Text content returned by the API, or (text, usage_dict) if return_usage=True
        """
        for attempt in range(self.max_retries):
            try:
                params = {
                    "model": model,
                    "input": prompt,
                }
                
                # Add web search tool if enabled
                if use_web_search:
                    params["tools"] = [{"type": "web_search"}]
                 
                response = self.client.responses.create(**params)                # Extract text
                output_text = None
                if hasattr(response, "output_text") and response.output_text is not None:
                    output_text = response.output_text
                else:
                    # Fallback: assemble text from response output if output_text missing
                    output_segments = []
                    for item in getattr(response, "output", []) or []:
                        if getattr(item, "type", None) == "output_text":
                            output_segments.append(getattr(item, "text", ""))
                        elif hasattr(item, "content"):
                            for content in item.content:
                                if getattr(content, "type", None) == "text":
                                    output_segments.append(getattr(content, "text", ""))

                    if output_segments:
                        output_text = "".join(output_segments)

                if output_text is None:
                    raise RuntimeError("No text returned in response output")

                if use_web_search:
                    # logger.info(f"Raw output with web search links:\n{output_text}")
                    output_text = self.remove_web_links(output_text)

                # Extract usage information if requested
                if return_usage:
                    usage_dict = None
                    if hasattr(response, "usage") and response.usage is not None:
                        usage_dict = {
                            "input_tokens": getattr(response.usage, "input_tokens", 0),
                            "output_tokens": getattr(response.usage, "output_tokens", 0),
                            "total_tokens": getattr(response.usage, "total_tokens", 0)
                        }
                    
                    # Count web search calls if web search is enabled
                    if use_web_search:
                        web_search_count = 0
                        if hasattr(response, "output") and response.output is not None:
                            for item in response.output:
                                if getattr(item, "type", None) == "web_search_call":
                                    web_search_count += 1
                        
                        # Add web search count to usage dict
                        if usage_dict is None:
                            usage_dict = {}
                        usage_dict["web_search_calls"] = web_search_count
                        logger.info(f"Web search was called {web_search_count} time(s)")
                    
                    return output_text, usage_dict
                else:
                    return output_text
                
            except openai.RateLimitError as e:
                wait_time = min(2 ** attempt, 60)  # exponential backoff, wait up to 60 seconds
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise e
                    
            except openai.APITimeoutError as e:
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"API timeout (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise e
                    
            except openai.APIConnectionError as e:
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"API connection error (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise e
                    
            except openai.APIStatusError as e:
                if e.status_code >= 500:  # Server errors, can be retried
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Server error {e.status_code} (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                        continue
                # Client errors (400-499) should not be retried
                raise e
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
        
        raise Exception(f"Failed after {self.max_retries} attempts")
    
    def simple_completion(self, prompt: str, model: str = "gpt-4o-mini", return_usage: bool = False, use_web_search: bool = False, **kwargs):
        """
        Simple text completion call
        
        Args:
            prompt: Input prompt text
            model: Model to use
            return_usage: If True, return tuple of (text, usage_dict), otherwise just text
            use_web_search: If True, enable web search tool for the API call
            **kwargs: Other parameters
            
        Returns:
            str or tuple: Generated text, or (text, usage_dict) if return_usage=True
        """
        # kwargs kept for backward compatibility but not forwarded to the API
        _ = kwargs
        return self.chat_completion(prompt=prompt, model=model, return_usage=return_usage, use_web_search=use_web_search)


# Convenience function using default configuration
def create_gpt_client(api_key: str = None, base_url: str = "https://api.openai.com/v1", max_retries: int = 3) -> GPTAPIClient:
    """
    Convenience function to create GPT API client
    
    Args:
        api_key: API key, if not provided will attempt to read from environment variable
        base_url: API base URL
        max_retries: Maximum number of retries
        
    Returns:
        GPTAPIClient: Client instance
    """
    if api_key is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
    
    return GPTAPIClient(api_key=api_key, base_url=base_url, max_retries=max_retries)