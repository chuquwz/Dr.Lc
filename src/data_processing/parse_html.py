"""
Data Processing Script for Dr.Lc RAG Chatbot
=============================================
Parse 1019 HTML files crawled from Long Châu pharmacy website.
Extract structured product data from Next.js __NEXT_DATA__ JSON.
Clean HTML tags from text fields.
Output: One JSON file per product in data-processed/ directory.

Usage:
    python src/data_processing/parse_html.py
"""

import sys
import io
import os
import json
import glob
import re
from bs4 import BeautifulSoup

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data-raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data-processed")


def clean_html(html_string: str) -> str:
    """Remove HTML tags from a string, keeping only clean text.
    
    Args:
        html_string: A string that may contain HTML tags.
        
    Returns:
        Clean text with HTML tags removed and whitespace normalized.
    """
    if not html_string:
        return ""
    soup = BeautifulSoup(html_string, "html.parser")
    
    # Remove all <a> tag href but keep text
    text = soup.get_text(separator="\n", strip=True)
    
    # Normalize whitespace: collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_product_data(html_content: str) -> dict | None:
    """Extract structured product data from a Long Châu HTML page.
    
    The HTML pages are Next.js SSR pages. All product data is embedded
    in a <script id="__NEXT_DATA__"> tag as JSON.
    
    Args:
        html_content: Raw HTML string of the product page.
        
    Returns:
        A dictionary with cleaned product data, or None if parsing fails.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find the Next.js data script
    next_data_script = soup.find("script", id="__NEXT_DATA__")
    if not next_data_script or not next_data_script.string:
        return None
    
    data = json.loads(next_data_script.string)
    page_props = data["props"]["pageProps"]
    product = page_props.get("product", {})
    content = page_props.get("content", {})
    faq_list = page_props.get("faq", [])
    
    # -----------------------------------------------------------
    # 1. Basic Info
    # -----------------------------------------------------------
    result = {
        "sku": product.get("sku", ""),
        "name": product.get("webName", ""),
        "short_name": product.get("shortName", ""),
        "official_name": product.get("officialProductName", ""),
        "url": page_props.get("url", ""),
        "image_url": product.get("primaryImage", {}).get("url", ""),
    }
    
    # -----------------------------------------------------------
    # 2. Classification
    # -----------------------------------------------------------
    categories = product.get("categories", [])
    result["categories"] = [
        {"level": cat["level"], "name": cat["name"]}
        for cat in categories
    ]
    
    result["brand"] = product.get("brand", "")
    result["manufacturer_country"] = product.get("manufactor", "")
    result["dosage_form"] = product.get("dosageForm", "")
    result["registration_number"] = product.get("registNum", "")
    result["expiration_date"] = product.get("expirationDate", "")
    
    # -----------------------------------------------------------
    # 3. Ingredients
    # -----------------------------------------------------------
    ingredients = product.get("ingredient", [])
    result["ingredients"] = [
        {"name": ing.get("name", ""), "amount": ing.get("shortDescription", "")}
        for ing in ingredients
    ]
    result["ingredient_per"] = product.get("ingredientFor", "")
    
    # -----------------------------------------------------------
    # 4. Pricing
    # -----------------------------------------------------------
    prices = product.get("prices", [])
    result["prices"] = [
        {
            "unit": p.get("measureUnitName", ""),
            "price": p.get("price", 0),
            "currency": p.get("currencySymbol", "đ"),
            "specs": p.get("productSpecs", ""),
        }
        for p in prices
    ]
    
    # -----------------------------------------------------------
    # 5. Medical Content (clean HTML → plain text)
    # -----------------------------------------------------------
    # Prefer content fields (more detailed), fallback to product fields
    result["description"] = clean_html(
        content.get("description") or product.get("description", "")
    )
    result["usage"] = clean_html(
        content.get("usage") or product.get("usage", "")
    )
    result["dosage"] = clean_html(
        content.get("dosage") or product.get("dosage", "")
    )
    result["side_effects"] = clean_html(
        content.get("adverseEffect") or product.get("adverseEffect", "")
    )
    result["precautions"] = clean_html(
        content.get("careful") or product.get("careful", "")
    )
    result["storage"] = clean_html(
        content.get("preservation") or product.get("preservation", "")
    )
    
    # -----------------------------------------------------------
    # 6. Warnings
    # -----------------------------------------------------------
    warnings = product.get("warning", [])
    result["warnings"] = [
        w.get("name", "") if isinstance(w, dict) else str(w)
        for w in warnings
        if (w.get("name") if isinstance(w, dict) else w)
    ]
    
    # -----------------------------------------------------------
    # 7. FAQ
    # -----------------------------------------------------------
    result["faq"] = [
        {
            "question": item.get("title", ""),
            "answer": clean_html(item.get("content", "")),
        }
        for item in faq_list
    ]
    
    return result


def process_all_files():
    """Process all HTML files in data-raw/ and save as JSON in data-processed/."""
    
    # Create output directory
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    html_files = glob.glob(os.path.join(DATA_RAW_DIR, "*.html"))
    print(f"Found {len(html_files)} HTML files to process.")
    
    success_count = 0
    fail_count = 0
    
    for i, filepath in enumerate(html_files, 1):
        filename = os.path.basename(filepath)
        file_id = filename.replace(".html", "")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            product_data = extract_product_data(html_content)
            
            if product_data is None:
                print(f"  [{i}/{len(html_files)}] SKIP: {filename} (no data found)")
                fail_count += 1
                continue
            
            # Save as JSON
            output_path = os.path.join(DATA_PROCESSED_DIR, f"{file_id}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(product_data, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            
            # Progress update every 100 files
            if i % 100 == 0 or i == len(html_files):
                print(f"  [{i}/{len(html_files)}] Processed... ({success_count} OK, {fail_count} failed)")
                
        except Exception as e:
            print(f"  [{i}/{len(html_files)}] ERROR: {filename} - {e}")
            fail_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"DONE!")
    print(f"  Total files:  {len(html_files)}")
    print(f"  Success:      {success_count}")
    print(f"  Failed:       {fail_count}")
    print(f"  Output dir:   {DATA_PROCESSED_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    process_all_files()
