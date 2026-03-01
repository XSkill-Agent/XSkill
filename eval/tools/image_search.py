"""
ImageSearch Tool - Use Serper.dev Google Images Search API for image search
Support text query image search and reverse image search (image search by image).
"""

import os
import json
import requests
from io import BytesIO
from PIL import Image
from tools.base import BaseTool
from tools.tool_registry import register_tool


@register_tool("image_search")
class ImageSearch(BaseTool):
    name = "image_search"
    description = '''
Search for related images using text query or reverse image search.
- For text-to-image search: specify search_type="text" and provide a query.
- For reverse image search: specify search_type="reverse" and provide an image_url.
Returns images with URLs and descriptions.

Notice:
To perform reverse image search, specify the image name directly.
- "original_image" - use the original input image
- "tool_image_N" - use tool-generated image N (from tool outputs, e.g., tool_image_1, tool_image_2)
- "observation_N" - use zoomed image regions from earlier zoom operations (e.g., observation_1, observation_2)
- Image URLs: Direct URL string (http:// or https://) from previous search results or other sources

You only have limited search times, so please use it wisely.
'''
    parameters = {
        "type": "object",
        "properties": {
            "search_type": {
                "type": "string",
                "enum": ["text", "reverse"],
                "description": "Type of search: 'text' for text-to-image search, 'reverse' for reverse image search",
                "default": "text"
            },
            "query": {
                "type": "string",
                "description": "Search query string (required for text search)"
            },
            "image_url": {
                "type": "string",
                "description": "Image filename, local reference, or URL (required for reverse search). Use 'original_image', 'tool_image_N', 'observation_N', or a direct image URL (http://... or https://...)"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of image results to return (default: 10)",
                "default": 10
            }
        },
        "required": []  # query or image_url is required
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Get API key from environment variable
        self.api_key = os.getenv("SERPAPI_KEY")
        
        # Get from config (if provided)
        if config and 'api_key' in config and config['api_key']:
            self.api_key = config['api_key']
        
        if not self.api_key:
            raise ValueError(
                "Serper.dev API key not found. Please set SERPAPI_KEY "
                "environment variable or provide api_key in config"
            )
        
        self.max_results_default = config.get('max_results', 10) if config else 10
        self.search_type_default = config.get('search_type', 'text') if config else 'text'
        self.api_endpoint = "https://google.serper.dev/images"  # Text image search endpoint
        self.reverse_image_endpoint = "https://google.serper.dev/lens"  # Reverse image search endpoint (using Serper.dev /lens API)
        self.timeout = config.get('timeout', 100) if config else 100  # Increase timeout, because reverse image search may take longer
        self.download_image_counter = 0  # Used to generate unique image file names
        
        # Search image compression configuration (get from config, if not provided, use default value)
        # Get args from config (if passed via api_tool_handler, args is Namespace object)
        args_obj = config.get('args') if config and isinstance(config, dict) else None
        if args_obj:
            # args is Namespace object, use getattr to access properties
            global_max_pixels = getattr(args_obj, 'max_pixels', 2000000)
            global_min_pixels = getattr(args_obj, 'min_pixels', 40000)
        else:
            global_max_pixels = 2000000
            global_min_pixels = 40000
        
        # Search image using more strict compression parameters (configurable, default is half of global)
        if config and isinstance(config, dict):
            self.search_image_max_pixels = config.get('search_image_max_pixels', global_max_pixels // 2)
            self.search_image_quality = config.get('search_image_quality', 75)
        else:
            self.search_image_max_pixels = global_max_pixels // 2
            self.search_image_quality = 75
        self.search_image_min_pixels = global_min_pixels  # Use global min_pixels
        
        # Proxy configuration (read from environment variables)
        self.proxies = None
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if http_proxy or https_proxy:
            self.proxies = {
                "http": http_proxy,
                "https": https_proxy or http_proxy
            }
            print(f"[ImageSearch] Using proxy: {self.proxies}")
    
    def _upload_local_image(self, image_path, max_retries=3, timeout=30):
        """
        Upload local image to image hosting service
        支持多个图床服务，按优先级自动切换：
        1. ImgBB (https://api.imgbb.com) - Requires API key, stable and reliable
        2. cloudflareimg.cdn.sn - Supports webp compression, no registration required
        3. 0x0.st (https://0x0.st) - Anonymous file upload service, no registration required, stable and reliable
        4. catbox.moe (https://catbox.moe) - Anonymous image upload service, no registration required, backup solution
        
        Support automatic retry mechanism, automatically switch to next service on failure

        Args:
            image_path: Local image path
            max_retries: Maximum number of retries for each service
            timeout: Timeout in seconds
            
        Returns:
            Image URL, None if failed
        """
        print(f"[ImageSearch] Uploading local image: {image_path}")
        
        def parse_0x0_response(response):
            """Parse 0x0.st response (return pure text URL)"""
            try:
                url = response.text.strip()
                if url.startswith('http://') or url.startswith('https://'):
                    return url
            except:
                pass
            return None
        
        def parse_catbox_response(response):
            """Parse catbox.moe response (return pure text URL)"""
            try:
                url = response.text.strip()
                if url.startswith('http://') or url.startswith('https://'):
                    return url
            except:
                pass
            return None
        
        def parse_imgbb_response(response):
            """Parse ImgBB response (JSON format)"""
            try:
                result = response.json()
                if result.get('success'):
                    return result.get('data', {}).get('url')
            except:
                pass
            return None
        
        def parse_cloudflareimg_response(response):
            """Parse cloudflareimg.cdn.sn response (JSON format)"""
            try:
                result = response.json()
                if result.get('success'):
                    uploaded_url = result.get('url')
                    # Display compression information (if available)
                    if 'data' in result:
                        compression_ratio = result['data'].get('compression_ratio', 0)
                        if compression_ratio:
                            print(f"[ImageSearch] Compression ratio: {compression_ratio}%")
                    return uploaded_url
            except:
                pass
            return None
        
        # Define multiple image hosting service configurations, sorted by priority
        upload_services = []
        
        # 1. Try ImgBB (if API key is provided, more stable and reliable)
        imgbb_key = os.getenv('IMGBB_API_KEY') or (self.config.get('imgbb_api_key') if hasattr(self, 'config') and self.config else None)
        if imgbb_key:
            upload_services.append({
                'name': 'ImgBB',
                'url': 'https://api.imgbb.com/1/upload',
                'files_key': 'image',  # ImgBB uses image as file field name
                'parse_response': parse_imgbb_response,
                'extra_data': {'key': imgbb_key}  # ImgBB requires API key
            })
        
        # 2. cloudflareimg.cdn.sn (supports webp compression, no API key required)
        upload_services.append({
            'name': 'cloudflareimg.cdn.sn',
            'url': 'https://cloudflareimg.cdn.sn/api/v1.php',
            'files_key': 'image',  # cloudflareimg.cdn.sn uses image as file field name
            'parse_response': parse_cloudflareimg_response,
            'extra_data': {'outputFormat': 'webp'}  # Use webp format for better compression rate
        })
        
        # 3. Anonymous service (no API key required)
        upload_services.extend([
            {
                'name': '0x0.st',
                'url': 'https://0x0.st',
                'files_key': 'file',  # 0x0.st uses file as file field name, return pure text URL
                'parse_response': parse_0x0_response
            },
            {
                'name': 'catbox.moe',
                'url': 'https://catbox.moe/user/api.php',
                'files_key': 'fileToUpload',  # catbox.moe uses fileToUpload as file field name, return pure text URL
                'parse_response': parse_catbox_response,
                'extra_data': {'reqtype': 'fileupload'}  # catbox.moe requires additional parameters
            }
        ])
        
        # Try each image hosting service in order
        for service in upload_services:
            print(f"[ImageSearch] Trying {service['name']}...")
            
            for attempt in range(max_retries):
                try:
                    with open(image_path, 'rb') as f:
                        filename = os.path.basename(image_path)
                        
                        # Build file dictionary
                        files = {service['files_key']: (filename, f, 'image/jpeg')}
                        
                        # Build additional data parameters
                        data = service.get('extra_data', {}).copy()
                        
                        response = requests.post(
                            service['url'],
                            files=files,
                            data=data,
                            timeout=timeout,
                            proxies=self.proxies if hasattr(self, 'proxies') else None
                        )
                    
                    if response.status_code == 200:
                        # Parse response to get image URL
                        uploaded_url = service['parse_response'](response)
                        
                        if uploaded_url:
                            print(f"[ImageSearch] Successfully uploaded to {service['name']}: {uploaded_url}")
                            return uploaded_url
                        else:
                            print(f"[ImageSearch] {service['name']} upload failed: Invalid response format - {response.text[:200]}")
                    else:
                        print(f"[ImageSearch] {service['name']} upload failed with status {response.status_code}: {response.text[:200]}")
                        
                except requests.exceptions.Timeout:
                    print(f"[ImageSearch] {service['name']} upload timeout (attempt {attempt + 1}/{max_retries})")
                except Exception as e:
                    print(f"[ImageSearch] {service['name']} upload error (attempt {attempt + 1}/{max_retries}): {e}")
                
                # If not the last attempt, wait a bit and try again
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)  # Wait 2 seconds to avoid triggering frequency limit
            
            # Current service failed, try next service
            print(f"[ImageSearch] {service['name']} failed, trying next service...")
        
        print(f"[ImageSearch] All upload services failed for {image_path}")
        return None
    
    def _download_image(self, image_url, save_dir, max_retries=3, timeout=10):
        """
        Download image and save to local
        
        Args:
            image_url: Image URL
            save_dir: Save directory
            max_retries: Maximum number of retries
            timeout: Timeout in seconds
            
        Returns:
            Local file path, None if failed
        """
        if not save_dir:
            return None
        
        # Generate file name
        self.download_image_counter += 1
        filename = f"search_image_{self.download_image_counter}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        print(f"[ImageSearch] Downloading {image_url[:80]}...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    image_url, 
                    headers=headers, 
                    timeout=timeout, 
                    stream=True,
                    proxies=self.proxies if hasattr(self, 'proxies') else None
                )
                if response.status_code == 200:
                    # Verify if it is a valid image
                    img_data = BytesIO(response.content)
                    img = Image.open(img_data)
                    img.verify()
                    
                    # Reload (after verify, need to reopen)
                    img_data.seek(0)
                    img = Image.open(img_data)
                    
                    # Convert to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Use process_image for compression (more strict parameters for search images)
                    from utils.context_utils import process_image
                    original_size = img.width * img.height
                    compressed_img = process_image(img, self.search_image_max_pixels, self.search_image_min_pixels)
                    compressed_size = compressed_img.width * compressed_img.height
                    
                    # Save compressed image (use lower quality to further reduce file size)
                    compressed_img.save(filepath, 'JPEG', quality=self.search_image_quality)
                    
                    # Record compression information
                    if original_size != compressed_size:
                        print(f"[ImageSearch] Downloaded and compressed: {filename} ({img.width}x{img.height} -> {compressed_img.width}x{compressed_img.height})")
                    else:
                        print(f"[ImageSearch] Downloaded: {filename} ({compressed_img.width}x{compressed_img.height})")
                    
                    return filepath
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[ImageSearch] Failed to download image: {e}")
        
        return None
    
    def call(self, params, **kwargs):
        """
        Execute image search (text search or reverse image search)
        
        Args:
            params: Dictionary containing search_type, query or image_url
            **kwargs: Additional parameters, can contain save_dir for downloading images
            
        Returns:
            Formatted search result string
        """
        # 解析参数
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                # If parsing fails, assume it is query
                params = {"query": params, "search_type": "text"}
        
        search_type = params.get("search_type", self.search_type_default)
        max_results = params.get("max_results", self.max_results_default)
        save_dir = kwargs.get("save_dir", None)
        
        # Standardize search_type
        if search_type in ["image", "reverse"]:
            search_type = "reverse"
        
        if search_type == "text":
            return self._text_to_image_search(params, max_results, save_dir)
        elif search_type == "reverse":
            return self._reverse_image_search(params, max_results, save_dir)
        else:
            return f"Error: Invalid search_type '{search_type}'. Must be 'text' or 'reverse'"
    
    def _text_to_image_search(self, params, max_results, save_dir=None):
        """
        Text query image
        
        Args:
            params: Dictionary containing query
            max_results: Maximum number of results
            save_dir: Save directory (if provided, download images)
            
        Returns:
            Formatted search result
        """
        query = params.get("query", "")
        if not query:
            return "Error: No query provided for text search"
        
        try:
            print(f"[ImageSearch] Text search for images: {query}")
            
            # Prepare request
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": min(max_results, 100)  # Serper.dev supports up to 100 results
            }
            
            # Send request (supports proxy, with retry mechanism)
            max_retries = 3
            retry_delay = 1
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                        proxies=self.proxies
                    )
                    break  # If successful, exit retry loop
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt < max_retries - 1:
                        print(f"[ImageSearch] Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        error_msg = f"Connection failed after {max_retries} attempts: {str(e)}. Please check network connection and DNS settings."
                        print(f"[ImageSearch] {error_msg}")
                        return f"Error: {error_msg}"
            
            if response is None:
                return "Error: Failed to get response from API"
            
            # Check response status
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}: {response.text[:200]}"
                print(f"[ImageSearch] {error_msg}")
                return f"Error: {error_msg}"
            
            # Parse response
            try:
                result_data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON response: {str(e)}. Response: {response.text[:200]}"
                print(f"[ImageSearch] {error_msg}")
                return f"Error: {error_msg}"
            
            # Extract image search results
            images = result_data.get("images", [])
            
            if not images:
                # Check if there is error information
                error_info = result_data.get("error", "")
                if error_info:
                    return f"Error: {error_info}"
                return f"No images found for query: '{query}'"
            
            # Limit result number
            images = images[:max_results]
            
            formatted = []
            
            for i, img in enumerate(images, 1):
                title = img.get('title', 'No title')
                image_url = img.get('imageUrl') or img.get('image_url') or img.get('link', 'N/A')
                source_url = img.get('link') or img.get('source') or img.get('sourceUrl', 'N/A')
                
                # Note: Images are not downloaded to local storage
                entry = f"Image: {image_url}, Text: {title}, Webpage Url: {source_url}"
                formatted.append(entry)
            
            output = "```\n" + '\n\n'.join(formatted) + "\n```"
            print(f"[ImageSearch] Found {len(images)} image results (returning URLs only, no markdown images)")
            return output
            
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            print(f"[ImageSearch] {error_msg}")
            return f"Error: {error_msg}"
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"[ImageSearch] {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error during text image search: {str(e)}"
            print(f"[ImageSearch] {error_msg}")
            import traceback
            traceback.print_exc()
            return f"Error: {error_msg}"
    
    def _reverse_image_search(self, params, max_results, save_dir=None):
        """
        Reverse image search (image search by image)
        
        Args:
            params: Dictionary containing image_url
            max_results: Maximum number of results
            save_dir: Save directory (if provided, download images)
            
        Returns:
            Formatted search result
        """
        image_url = params.get("image_url", "")
        if not image_url:
            return "Error: No image_url provided for reverse search"
        
        try:
            # Check if it is a local file path
            original_image_path = image_url
            
            # Check if it is a local file path (exclude already URL or data URI cases)
            if not image_url.startswith('http://') and not image_url.startswith('https://') and not image_url.startswith('data:'):
                # Maybe it is a local file path, check if file exists
                if os.path.exists(image_url) or os.path.isfile(image_url):
                    print(f"[ImageSearch] Detected local file path: {image_url}")
                    # Upload local image to image hosting service
                    uploaded_url = self._upload_local_image(image_url)
                    if uploaded_url:
                        image_url = uploaded_url
                        print(f"[ImageSearch] Using uploaded image URL for reverse search: {uploaded_url}")
                    else:
                        return f"Error: Failed to upload local image file {image_url} to image hosting service. Please provide a public image URL instead."
            
            print(f"[ImageSearch] Reverse image search for: {image_url[:100]}..." if len(image_url) > 100 else f"[ImageSearch] Reverse image search for: {image_url}")
            
            # Prepare request
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Use Serper.dev /lens endpoint for reverse image search
            # Based on test code, use simple payload format: {"url": image_url, "num": max_results}
            payload = {
                "url": image_url,
                "num": min(max_results, 100)  # Limit result number
            }
            
            print(f"[ImageSearch] Sending reverse image search request to Serper.dev /lens")
            print(f"[ImageSearch] Image URL: {image_url[:80]}...")
            
            # Send request (supports proxy, with retry mechanism)
            # Note: Use data=json.dumps(payload) instead of json=payload, consistent with test code
            max_retries = 3
            retry_delay = 1
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.reverse_image_endpoint,
                        headers=headers,
                        data=json.dumps(payload),  # Use data instead of json, and manually serialize
                        timeout=self.timeout,
                        proxies=self.proxies
                    )
                    
                    # If successful, exit retry loop
                    if response.status_code == 200:
                        break
                    
                    # If not the last attempt, wait and try again
                    if attempt < max_retries - 1:
                        error_text = response.text[:200] if hasattr(response, 'text') else str(response)
                        print(f"[ImageSearch] Request failed (attempt {attempt + 1}/{max_retries}): status {response.status_code}, {error_text}, retrying...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    
                    break  # Exit retry loop
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt < max_retries - 1:
                        print(f"[ImageSearch] Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        error_msg = f"Connection failed after {max_retries} attempts: {str(e)}. Please check network connection and DNS settings."
                        print(f"[ImageSearch] {error_msg}")
                        return f"Error: {error_msg}"
            
            if response is None:
                return "Error: Failed to get response from API"
            
            # Check response status
            if response.status_code != 200:
                error_text = response.text[:500] if hasattr(response, 'text') else str(response)
                error_msg = f"API request failed with status {response.status_code}: {error_text}"
                print(f"[ImageSearch] {error_msg}")
                return f"Error: {error_msg}"
            
            # Parse response
            try:
                result_data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON response: {str(e)}. Response: {response.text[:200]}"
                print(f"[ImageSearch] {error_msg}")
                return f"Error: {error_msg}"
            
            # Serper.dev /lens endpoint returns data structure:
            # - organic: Search results array (main format)
            #   Each result contains: title, source, link, imageUrl, thumbnailUrl
            # - credits: Remaining credits
            # - searchParameters: Search parameters
            
            # Extract all types of matching results
            all_results = []
            
            # Prioritize processing Serper.dev's organic array format
            organic = result_data.get("organic", [])
            if organic:
                print(f"[ImageSearch] Found {len(organic)} organic results from Serper.dev")
                all_results = organic[:max_results]
            
            # If still no results, try other formats (compatible with SerpAPI format)
            if not all_results:
                # Visual matching (SerpAPI format)
                visual_matches = result_data.get("visual_matches", [])
                if visual_matches:
                    print(f"[ImageSearch] Found {len(visual_matches)} visual matches")
                    all_results.extend(visual_matches[:max_results])
                
                # Exact matching (SerpAPI format)
                exact_matches = result_data.get("exact_matches", [])
                if exact_matches and len(all_results) < max_results:
                    print(f"[ImageSearch] Found {len(exact_matches)} exact matches")
                    remaining = max_results - len(all_results)
                    all_results.extend(exact_matches[:remaining])
                
                # Product results (SerpAPI format)
                products = result_data.get("products", [])
                if products and len(all_results) < max_results:
                    print(f"[ImageSearch] Found {len(products)} product matches")
                    remaining = max_results - len(all_results)
                    all_results.extend(products[:remaining])
                
                # If still no results, try from images field (compatible with old format)
                if not all_results:
                    images = result_data.get("images", [])
                    if images:
                        print(f"[ImageSearch] Found {len(images)} image results (legacy format)")
                        all_results = images[:max_results]
            
            if not all_results:
                # Check if there is error information
                error_info = result_data.get("error", "")
                if error_info:
                    return f"Error: {error_info}"
                # Display simplified prompt
                display_url = original_image_path if 'original_image_path' in locals() and original_image_path != image_url else image_url
                return f"No matches found for reverse image search of: {display_url}"
            
            # Limit result number
            all_results = all_results[:max_results]
            
            # Format output (align with WebWatcher format: pure text URL, no markdown image reference)
            formatted = []
            
            for i, item in enumerate(all_results, 1):
                # 处理 Serper.dev 的响应格式
                title = item.get('title') or item.get('name') or 'No title'
                # Serper.dev 使用 imageUrl 和 thumbnailUrl
                image_url_result = item.get('imageUrl') or item.get('thumbnailUrl') or item.get('thumbnail') or item.get('image_url') or item.get('link', 'N/A')
                source_url = item.get('link') or item.get('source') or item.get('sourceUrl', 'N/A')
                source_name = item.get('source', 'Unknown')
                
                # 构建结果条目（WebWatcher 格式：纯文本 URL）
                # Note: Images are not downloaded to local storage
                entry = f"Image: {image_url_result}, Text: {title}, Webpage Url: {source_url}"
                formatted.append(entry)
            
            # 使用 WebWatcher 的代码块格式
            output = "```\n" + '\n\n'.join(formatted) + "\n```"
            print(f"[ImageSearch] Found {len(all_results)} reverse search results (returning URLs only, no markdown images)")
            return output
            
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            print(f"[ImageSearch] {error_msg}")
            return f"Error: {error_msg}"
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"[ImageSearch] {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error during reverse image search: {str(e)}"
            print(f"[ImageSearch] {error_msg}")
            import traceback
            traceback.print_exc()
            return f"Error: {error_msg}"
