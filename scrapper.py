import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def generate_test_data(urls: list, company_id: str):
    output_dir = f"./data/{company_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating test data for {company_id}...")
    
    for i, url in enumerate(urls):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.extract()
                
            clean_text = soup.get_text(separator='\n', strip=True)
            
            # The exact schema the scraping team will provide
            payload = {
                "url": url,
                "scraped_at": datetime.now().isoformat(),
                "text": clean_text
            }
            
            filepath = os.path.join(output_dir, f"source_{i+1:03d}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
                
            print(f"Saved: {url} -> {filepath}")
            
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

if __name__ == "__main__":
    # Test URLs for a company
    test_urls = [
        "https://www.leadsquared.com/about/",
        "https://techcrunch.com/2022/06/21/daily-crunch-with-153m-series-c-leadsquared-becomes-indias-newest-unicorn/",
        "https://www.leadsquared.com/careers/",
        "https://help.leadsquared.com/news/universal-data-sync-uds-update-march-26/"
    ]
    generate_test_data(test_urls, "leadsquared")