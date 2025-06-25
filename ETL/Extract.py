import os
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re
def extract(headers) -> pd.DataFrame:
    """
    Extracts review data from British Airways reviews on AirlineQuality.com.

    Args:
        number_of_pages (int): The number of pages to scrape.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted review data.
    """
    base_url = "https://www.airlinequality.com/airline-reviews/united-airlines"
    page_size = 100

    reviews_data = []
    # Lấy số trang cần quét
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    pagination_div = soup.find('div', class_='pagination-total').get_text(strip=True)
    numbers = re.findall(r'\d+', pagination_div)
    total_reviews = int(numbers[-1])
    number_pages= total_reviews // page_size + 1

    # Bắt đầu quét từ trang đầu tiên cho đến số trang tối đa
    for page in range(1, number_pages + 1):
        logging.info(f"Scraping page {page}")

        url = f"{base_url}/page/{page}/?sortby=post_date%3ADesc&pagesize={page_size}"
        print(url)
        try:
            response = requests.get(url,headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch page {page}: {e}")
            continue

        parsed_content = BeautifulSoup(response.content, 'html.parser')
        reviews = parsed_content.select('article[class*="comp_media-review-rated"]')

        for review in reviews:
            review_data = extract_review_data(review)
            reviews_data.append(review_data)

    df = pd.DataFrame(reviews_data)
    return df

def extract_review_data(review: BeautifulSoup) -> Dict[str, str]:
    """
    Extracts relevant data from a single review.

    Args:
        review (BeautifulSoup): The parsed HTML of a single review.

    Returns:
        Dict[str, str]: A dictionary containing extracted review data.
    """
    review_data = {
        'dates': extract_text(review, "time", itemprop="datePublished"),
        'customer_names': extract_text(review, "span", itemprop="name"),
        'countries': extract_country(review),
        'review_bodies': extract_text(review, "div", itemprop="reviewBody")
    }

    extract_ratings(review, review_data)

    return review_data

def extract_text(element: BeautifulSoup, tag: str, **attrs) -> str:
    """
    Extracts text from a BeautifulSoup element.

    Args:
        element (BeautifulSoup): The BeautifulSoup element to search within.
        tag (str): The HTML tag to look for.
        **attrs: Additional attributes to filter the search.

    Returns:
        str: The extracted text, or None if not found.
    """
    found = element.find(tag, attrs)
    return found.text.strip() if found else None

def extract_country(review: BeautifulSoup) -> str:
    """
    Extracts the country from a review.

    Args:
        review (BeautifulSoup): The parsed HTML of a single review.

    Returns:
        str: The extracted country, or None if not found.
    """
    country = review.find(string=lambda string: string and '(' in string and ')' in string)
    return country.strip('()') if country else None

def extract_ratings(review: BeautifulSoup, review_data: Dict[str, str]) -> None:
    """
    Extracts ratings from a review and adds them to the review_data dictionary.

    Args:
        review (BeautifulSoup): The parsed HTML of a single review.
        review_data (Dict[str, str]): The dictionary to update with extracted ratings.
    """
    review_ratings = review.find('table', class_='review-ratings')
    if not review_ratings:
        return

    for row in review_ratings.find_all('tr'):
        header = row.find('td', class_='review-rating-header')
        if not header:
            continue

        header_text = header.text.strip()
        if header_text in ['Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service', 'Wifi & Connectivity', 'Value For Money']:
            stars_td = row.find('td', class_='review-rating-stars')
            if stars_td:
                stars = stars_td.find_all('span', class_='star fill')
                review_data[header_text] = len(stars)
        else:
            value = row.find('td', class_='review-value')
            if value:
                review_data[header_text] = value.text.strip()

def main():
    headers = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
    
    extract(headers)
if __name__ == "__main__":
    df= main()