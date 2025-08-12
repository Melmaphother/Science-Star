import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, quote_plus, parse_qs
import os
import json
import re
import undetected_chromedriver as uc

def search_google(query, num_results=10, proxy=None):
    """
    Performs a Google search using an undetected chromedriver to avoid bot detection.
    
    Args:
        query (str): The search term.
        num_results (int): The number of results to retrieve.
        proxy (str, optional): Proxy server to use (e.g., "http://user:pass@host:port"). Defaults to None.

    Returns:
        list: A list of dictionaries, each containing search result data.
    """
    encoded_query = quote_plus(query)
    search_url = f"https://www.google.com/search?q={encoded_query}&num={num_results}&hl=en"

    print(f"[*] Searching Google for '{query}'...")

    options = uc.ChromeOptions()
    
    # Use a persistent user profile to save session data (cookies, etc.)
    profile_path = os.path.join(os.getcwd(), "chrome_profile")
    options.add_argument(f"--user-data-dir={profile_path}")
    
    if proxy:
        print(f"[*] Using proxy: {proxy}")
        options.add_argument(f'--proxy-server={proxy}')
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    
    driver = None
    try:
        # IMPORTANT: Set the version_main to your installed Chrome's major version.
        # e.g., if your Chrome is version 139.0.7258.67, use 139.
        driver = uc.Chrome(options=options, version_main=139)
        driver.get(search_url)

        # Check if manual intervention is needed for CAPTCHA or consent.
        if "google.com/sorry/" in driver.current_url or "consent.google.com" in driver.current_url:
            print("\n" + "="*50)
            print("[ACTION REQUIRED] The browser may need your attention.")
            print("Please complete any manual steps (like CAPTCHA) if they appear.")
            print("Once you see the normal search results, press Enter here to continue.")
            print("="*50 + "\n")
            input("Press Enter to continue...")

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        if "google.com/sorry/" in driver.current_url:
            print("[!] Blocked by Google's 'sorry' page. Try using a different proxy or wait a while.")
            return []

        search_results = []
        for container in soup.find_all('div'):
            title_tag = container.find('h3')
            link_tag = container.find('a')

            if title_tag and link_tag and link_tag.get('href'):
                title = title_tag.get_text()
                link = link_tag.get('href')

                if title not in link_tag.get_text():
                    continue
                
                if link.startswith("/url?q="):
                    try:
                        link = parse_qs(urlparse(link).query)['q'][0]
                    except (KeyError, IndexError):
                        continue

                if not link.startswith("http"):
                    continue
                
                snippet_tag = container.find('div', {'data-sncf': '1'})
                snippet = snippet_tag.get_text(separator=' ', strip=True) if snippet_tag else "No snippet available."

                if not any(r['link'] == link for r in search_results):
                    search_results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })

                if len(search_results) >= num_results:
                    break
        
        print(f"[+] Found {len(search_results)} results from Google.")
        return search_results

    except Exception as e:
        print(f"[!] An error occurred during the browser-based search: {e}")
        return []
    finally:
        if driver:
            driver.quit()


def scrape_page_content(url):
    """
    Scrapes the main content and metadata from a given webpage URL.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print(f"    [*] Scraping content from {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract main content from common semantic tags
        if soup.find('article'):
            content = soup.find('article').get_text(' ', strip=True)
        elif soup.find('main'):
            content = soup.find('main').get_text(' ', strip=True)
        else:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(' ', strip=True) for p in paragraphs])

        # Extract publication date
        date = None
        meta_date = soup.find('meta', attrs={'property': 'article:published_time'})
        if meta_date and 'content' in meta_date.attrs:
            date = meta_date['content']
        else:
            time_tag = soup.find('time')
            if time_tag and 'datetime' in time_tag.attrs:
                date = time_tag['datetime']
            elif time_tag:
                date = time_tag.get_text()
        
        # Extract the page's own description/snippet
        subpage_snippet = None
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and 'content' in meta_desc.attrs:
            subpage_snippet = meta_desc['content']

        parsed_url = urlparse(url)
        source = parsed_url.netloc

        return {
            "full_content": content,
            "date": date,
            "source": source,
            "subpage_snippet": subpage_snippet
        }

    except requests.exceptions.RequestException as e:
        print(f"    [!] Failed to scrape {url}. Reason: {e}")
        return None


def simulate_search_api(query, top_k=5, proxy=None):
    """
    Orchestrates the two-step process of searching and then scraping results.
    """
    google_results = search_google(query, num_results=top_k, proxy=proxy)

    if not google_results:
        print(f"[!] Could not retrieve initial search results for query: '{query}'. Skipping.")
        return []

    final_results = []
    for idx, result in enumerate(google_results):
        time.sleep(1) # Be polite to servers
        page_data = scrape_page_content(result['link'])

        if page_data and page_data["full_content"]:
            _search_result = {
                "idx": idx,
                "title": result["title"],
                "date": page_data["date"],
                "google_snippet": result["snippet"],
                "subpage_snippet": page_data["subpage_snippet"],
                "source": page_data["source"],
                "link": result['link'],
                "content": page_data["full_content"]
            }
            final_results.append(_search_result)
            print(f"    [+] Successfully processed result {idx+1}/{len(google_results)}")
        else:
            print(f"    [-] Skipping result {idx+1} due to scraping failure.")
    
    return final_results

def sanitize_filename(query):
    """
    Cleans a string to be used as a valid filename.
    """
    sanitized = re.sub(r'[\\/*?:"<>|]', "", query)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:100]


if __name__ == "__main__":
    # --- Configuration ---
    initial_query = "abc"
    queries_to_process = [
        "What is transformer in deep learning?",
        "Latest advancements in large language models",
        "Python undetected-chromedriver tutorial"
    ]
    
    number_of_results_to_process = 3
    output_directory = "search_outputs"
    
    # --- (Optional) Proxy Configuration ---
    # If you have a proxy, set it here. Otherwise, leave it as None.
    # Format: "http://user:password@host:port" or "http://host:port"
    proxy_server = None 
    
    # --- Main Execution Logic ---
    os.makedirs(output_directory, exist_ok=True)

    if not os.path.exists("chrome_profile"):
        print("\n" + "="*80)
        print("--- Chrome profile not found, try to initialize it ---")
        print("="*80 + "\n")

        simulate_search_api(initial_query, top_k=1, proxy=proxy_server)

        if os.path.exists("chrome_profile"):
            print("\n" + "="*80)
            print("--- Chrome profile initialized ---")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("--- Chrome profile initialization failed ---")
            print("="*80 + "\n")
            exit(1)

    for i, query in enumerate(queries_to_process):
        print("\n" + "="*80)
        print(f"--- Processing Query {i+1}/{len(queries_to_process)}: '{query}' ---")
        print("="*80 + "\n")
        
        final_data = simulate_search_api(query, top_k=number_of_results_to_process, proxy=proxy_server)

        print(f"\n--- Query Processing Complete for '{query}' ---")

        if final_data:
            print(f"\n[SUCCESS] Retrieved and processed {len(final_data)} results for this query.\n")
            
            output_filename = sanitize_filename(query) + ".jsonl"
            output_path = os.path.join(output_directory, output_filename)

            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in final_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"[+] Results for this query successfully saved to '{output_path}'")
            except IOError as e:
                print(f"[!] Error saving results to file: {e}")
        else:
            print(f"\n[FAILURE] No data was processed for the query: '{query}'.")

    print("\n\n--- All queries have been processed. ---")
