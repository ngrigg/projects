from scrapers import scraper_utils as s_utils, goog_search_scraper as g_scraper
from utils import db_manager as ngdb, misc_utils as m_utils
from celery_app import app
import re
from datetime import datetime, timedelta
import numpy as np
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException
from urllib.parse import unquote
from collections import Counter
import urllib.parse
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import traceback
nltk.download('stopwords')


def generate_google_search_queries(url):
    """
    Generates multiple Google search queries for a given URL.

    This function constructs several Google search queries using keywords extracted from the provided URL,
    with variations to target specific aspects of the Grubhub site.

    :param url: The URL to generate the search queries for.

    :return: A list of generated Google search queries.

    Called by: (Typically called as part of a larger data processing or scraping workflow)
    """
    parts = url.split("/restaurant/")[-1]
    # Remove numbers and split by non-alphabetical characters
    keywords = re.split(r'[^a-zA-Z]+', re.sub(r'\d+', '', parts))
    keywords_joined = " ".join(keywords)

    # List of queries
    search_queries = [
        f"intitle:{keywords_joined} grubhub brooklyn OR queens OR bronx OR manhattan OR \"new york\" -reviews",
        f"site:grubhub.com {keywords_joined}"
    ]
    return search_queries


@app.task
def get_gh_restaurant_data_from_google_cache_gh_restaurant_page(task_data):
    """
    Extracts various data points from a Grubhub restaurant page.

    This function navigates to the provided Grubhub page URL using a WebDriver and extracts
    information such as the restaurant's name, neighborhood, phone number, categories, price score,
    address, ratings facets, menu data, overall ratings, ratings count, and reviews.

    Parameters:
    - task_data: Dictionary containing necessary data for scraping.
    """
    driver = task_data['driver']
    session_id, executor_url = driver['session_id'], driver['executor_url']
    driver = s_utils.create_driver_session(session_id, executor_url)
    url = task_data['search_url']
    print(f"Starting get_gh_restaurant_data_from_google_cache_gh_restaurant_page for {url}")

    try:
        # Extracting necessary variables from task_data

        # Navigate to the URL
        driver.get(url)

        # Logging process initiation and updating task status
        ngdb.insert_one('logs', {'message': f"Started processing: {url}", 'timestamp': datetime.now()})
        ngdb.upsert_one('tasks', {'url': url}, {'url': url, 'status': 'pending'})

        print("Extracting top_summary_data")
        all_top_summary_content = get_raw_top_summary_content(driver)
        top_summary_data = clean_top_summary_data(all_top_summary_content, url) if all_top_summary_content else {
            "name_neighborhood": None,
            "categories": None,
            "dollar_sign_price_score": None,
            "address": None,
            "phone_number": None
        }
        print(f"Found top summary data {top_summary_data}")

        address = top_summary_data['address']

        ratings_facets_data = extract_ratings_facets(driver)
        menu_data = extract_menu_data(driver, url)
        overall_ratings = extract_overall_ratings(driver)
        reviews = extract_reviews(driver)

        # Process and combine the extracted data
        print("Sending data for processing by process_restaurant_data")
        processed_data = process_restaurant_data(driver, top_summary_data, address, ratings_facets_data, menu_data,
                                                 overall_ratings, reviews)

        # Update task status and save data if required
        task_data['scheduler'].update_database_task_status(task_data, 'completed')
        task_data['db'].save_extracted_data(processed_data)
        print(f"Completed processing URL: {url}")
        return

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        # Handle exceptions and update task status accordingly
        print("Exception", e)
        m_utils.take_screenshot(driver, url)
        print(f"Full traceback: {traceback.format_exc()}")
        task_data['scheduler'].requeue_task(task_data)
        return


def get_raw_top_summary_content(driver):
    """
    Extracts the top summary content from a Grubhub restaurant page.

    :param driver: Selenium WebDriver instance.
    :return: Dictionary with summary data like name, neighborhood, phone number, etc.
    """
    error_count = 0
    max_error_limit = 3

    # Selector chains for the top summary content
    top_summary_element_chain = [
        "#ghs-restaurant-about",
        "[data-testid='restaurant-about']",
        ".restaurantAbout"
    ]

    wait = WebDriverWait(driver, 20)

    print("Trying to find top element...")
    for selector in top_summary_element_chain:
        print(f"Trying to find {selector}")
        try:
            top_summary_element_present = ec.presence_of_element_located((By.CSS_SELECTOR, selector))
            wait.until(top_summary_element_present)
            top_summary_element = driver.find_element(By.CSS_SELECTOR, selector)

            if top_summary_element:
                top_summary_content = top_summary_element.text.split('\n')
                return top_summary_content
            else:
                print("did not find top_summary_element")
                return []

        except (TimeoutException, NoSuchElementException, WebDriverException) as e:
            print("error finding top_summary_element")
            print("Exception", e)
            print(f"Full traceback: {traceback.format_exc()}")
            error_count += 1
            if error_count >= max_error_limit:
                return []
            continue


def clean_top_summary_data(all_top_summary_content, url):
    print(f"Cleaning top summary data {all_top_summary_content} for {url}")
    name_neighborhood, dollar_sign_price_score, dollar_sign_price_score, address, categories, phone_number = None, None, None, None, None, None
    try:
        if "reviews" in url:
            phone_regex = r"\(\d{3}\) \d{3}-\d{4}"
            price_regex = r"\$\$?\$?\$?\$?"
            address_part1_regex = r"\d+.*"
            address_part2_regex = r".*, [A-Z]{2} \d{5}"
            cuisines = set(map(str.lower, m_utils.load_cuisines_from_file("all_cuisines.txt")))

            name_neighborhood = all_top_summary_content.pop(0)
            dollar_sign_price_score = None
            address_parts = []
            phone_number = None
            price_count = 0

            # Using a while loop to safely remove items
            i = 0
            while i < len(all_top_summary_content):
                item = all_top_summary_content[i]
                if re.match(phone_regex, item) and not phone_number:
                    phone_number = item
                    all_top_summary_content.pop(i)
                    continue  # Skip incrementing i as the current item is removed
                elif re.match(price_regex, item):
                    price_count += 1
                    if price_count == 2:  # Capture the second occurrence
                        dollar_sign_price_score = item
                        all_top_summary_content.pop(i)
                        continue
                elif re.match(address_part1_regex, item) or re.match(address_part2_regex, item):
                    address_parts.append(item)
                    all_top_summary_content.pop(i)
                    continue
                i += 1

            # Process categories
            categories = [item.strip(',').lower() for item in all_top_summary_content if
                          item.strip(',').lower() in cuisines]
            address = ', '.join(address_parts) if address_parts else None
        else:
            name_neighborhood = all_top_summary_content[0] if len(all_top_summary_content) > 0 else None
            categories_str = all_top_summary_content[1] if len(all_top_summary_content) > 1 else None
            categories = set(categories_str.split(","))
            dollar_sign_price_score = all_top_summary_content[3] if len(all_top_summary_content) > 3 else None
            address = all_top_summary_content[4] + ", " + all_top_summary_content[5] if len(all_top_summary_content) > 5 else None
            phone_number = all_top_summary_content[6] if len(all_top_summary_content) > 6 else None

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")

    top_summary_data = {
        "name_neighborhood": name_neighborhood,
        "categories": categories,
        "dollar_sign_price_score": dollar_sign_price_score,
        "address": address,
        "phone_number": phone_number,
    }

    print(f"found top summary data: {top_summary_data}")

    return top_summary_data


def extract_ratings_facets(driver):
    """
    Extracts ratings facets from a Grubhub restaurant page.

    :param driver: Selenium WebDriver instance.
    :return: Dictionary with ratings facets.
    """
    error_count = 0
    max_error_limit = 2
    ratings_facets = {}

    try:
        ratings_facets_container = driver.find_element(By.CSS_SELECTOR, "div.ratingsFacets")
        facets = ratings_facets_container.find_elements(By.CSS_SELECTOR, "li.ratingsFacet-facetList-listItem")

        for facet in facets:
            percentage = s_utils.get_element_text_or_default_chain(facet, [[".ratingsFacet-percent"]], default=0)
            description = s_utils.get_element_text_or_default_chain(facet, [["span[color='secondaryText']"]], default="")
            key = description.lower().replace(" ", "_").replace("-", "_")
            try:
                value = int(percentage.rstrip('%'))
            except Exception as e:
                print("Exception", e)
                print(f"Full traceback: {traceback.format_exc()}")
                value = 0
            ratings_facets[key] = value

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        error_count += 1
        if error_count >= max_error_limit:
            return {}

    print(f"found ratings facets: {ratings_facets}")
    return ratings_facets


def extract_menu_data(driver, url):
    """
    Extracts menu data from a Grubhub restaurant page.

    :param driver: Selenium WebDriver instance.
    :param url: URL of the restaurant
    :return: List of dictionaries, each representing a menu item.
    """
    error_count = 0
    max_error_limit = 4
    menu_data = []

    if "reviews" in url:
        print("No menu because this is a reviews page. Skipping menu extraction.")
        return None

    try:
        # Wait for menu items to be visible
        wait = WebDriverWait(driver, 20)
        wait.until(ec.visibility_of_element_located((By.CSS_SELECTOR, "div.menuItem-group.u-flex")))
        menu_categories = driver.find_elements(By.CSS_SELECTOR, "div.menuItem-group.u-flex")

        for menu_category in menu_categories:
            items = menu_category.find_elements(By.CSS_SELECTOR, "div:nth-child(1)")
            for item in items:
                try:
                    item_name = s_utils.get_element_text_or_default_chain(item, [["div.menuItemNew-name"]], default="")
                    item_price = s_utils.get_element_text_or_default_chain(item, [["div.menuItemNew-price > h6 > span"]],
                                                                           default=0)
                    item_is_bestseller = bool(item.find_elements(By.XPATH, ".//article/button/div/div[1]/span"))

                    menu_data.append({
                        "item_name": item_name,
                        "item_price": item_price,
                        "item_is_bestseller": item_is_bestseller
                    })
                except Exception as e:
                    print("Exception", e)
                    print(f"Full traceback: {traceback.format_exc()}")
                    pass

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        error_count += 1
        if error_count >= max_error_limit:
            return []

    return menu_data


def extract_overall_ratings(driver):
    """
    Extracts overall ratings and ratings count from a Grubhub restaurant page.

    :param driver: Selenium WebDriver instance.
    :return: Tuple containing a list of overall ratings and the ratings count.
    """
    overall_ratings = []

    try:
        # Extracting overall ratings
        overall_rating_elements = driver.find_elements(By.CSS_SELECTOR, 'span[data-testid="single-star-text"]')
        for rating_element in overall_rating_elements:
            overall_ratings.append(rating_element.text)

        # Extracting ratings count
        try:
            ratings_count_element = driver.find_element(By.CSS_SELECTOR, 'span[data-testid="star-rating-text"]')
            ratings_text = str(ratings_count_element.text) if ratings_count_element.text else ""
            ratings_count = ''.join(filter(str.isdigit, ratings_text))
        except (TimeoutException, NoSuchElementException, WebDriverException) as e:
            print("Exception", e)
            print(f"Full traceback: {traceback.format_exc()}")
            ratings_count = None

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return [], None

    return overall_ratings, ratings_count


def extract_reviews(driver):
    """
    Extracts reviews from a Grubhub restaurant page.

    :param driver: Selenium WebDriver instance.
    :return: List of dictionaries, each containing review text and date.
    """
    reviews = []

    try:
        review_elements = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="restaurant-review-item"]')
        for review in review_elements:
            review_text = review.find_element(By.CSS_SELECTOR, 'span[data-testid="review-content"]').text
            review_date_text = review.find_element(By.XPATH, './div/div[1]/div[2]/div[1]/span').text
            review_date = convert_date(review_date_text)  # convert_date needs to be defined
            reviews.append({"text": review_text, "date": review_date})

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return []

    return reviews


def process_restaurant_data(driver, top_summary_data, address, ratings_facets, menu_data, overall_ratings, reviews):
    """
    Processes and combines restaurant data from various extracted components.

    :param driver: selenium webdriver
    :param top_summary_data: Extracted top summary data.
    :param address: Address from scraped restaurant data
    :param ratings_facets: Extracted ratings facets data.
    :param menu_data: Extracted menu data.
    :param overall_ratings: Extracted overall ratings
    :param reviews: Extracted reviews

    :return: Combined and processed restaurant data.
    """
    processed_data = {}
    try:
        # Combining data from different sources
        processed_data.update(top_summary_data)
        processed_data.update({"ratings_facets": ratings_facets})
        processed_data.update({"overall_ratings": overall_ratings})

        if menu_data:
            cleaned_menu_data = clean_and_process_menu_data(menu_data)
            processed_data.update({"cleaned_menu_data": cleaned_menu_data})
            common_words = calculate_common_words(cleaned_menu_data)
            processed_data['common_words'] = common_words
            processed_data['average_prices'] = calculate_average_prices(cleaned_menu_data, common_words)

        if reviews:
            processed_data.update({"reviews": reviews})
            processed_data['sentiment_analysis'] = analyze_reviews_sentiments(reviews)
            processed_data['avg_days_between_reviews'] = calculate_avg_days_between_reviews(reviews)
        if address:
            processed_data['zestimate'] = get_zestimate_from_address(driver, address)
        return processed_data
    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return processed_data


def get_zestimate_from_zillow_google_cache_url(url):
    """
    Extracts rent estimate values from a given Google cache URL.

    This function makes a web request to the provided URL and uses regex to find
    all instances of rent estimates on the page. It then filters these values based
    on context to ensure they are relevant to the rent estimate and not other numerical
    values like square footage. It handles multiple values by averaging them.

    :param url: The URL of the Google cached page to extract rent estimates from.

    :return: The average rent estimate if multiple values are found, a single value if only one is found,
             or None if no valid estimates are identified.
    """
    print(f"Extracting rent values for {url}")
    rent_regex = r"\$[1-9]\d{0,2}(?:,\d{3})*"
    valid_rent_values = []

    try:
        #     print("Requesting response")
        response = requests.get(url)
        if response.status_code == 200:
            #         print("Status code 200 (success)")
            page_source = response.text
            all_matches = re.finditer(rent_regex, page_source)

            for match in all_matches:
                start_index_narrow = max(0, match.start() - 50)
                narrow_context = page_source[start_index_narrow:match.end()]
                start_index_wide = max(0, match.start() - 300)
                wide_context = page_source[start_index_wide:match.end()]
                #             print(f"Testing match: {match}")

                # Check for 'estimate' in the context and ensure no unwanted substrings
                #             print("Testing match against surrounding context")
                if "estimate" in wide_context.lower() and not any(
                        substring in narrow_context.lower() for substring in ["sqft", "feet", "ft"]):
                    value = int(match.group().replace("$", "").replace(",", ""))
                    if 1000 <= value <= 10000:
                        #                     print(f"Appending value: {value}")
                        valid_rent_values.append(value)

        # Remove duplicates
        valid_rent_values = list(set(valid_rent_values))
        #     print(f"Possibly valid rent values: {valid_rent_values}")

        # Handle multiple values
        if len(valid_rent_values) > 1:
            #         print("Error: Multiple values found, taking average.")
            average_rent = sum(valid_rent_values) / len(valid_rent_values)
            return average_rent
        elif len(valid_rent_values) == 1:
            return valid_rent_values[0]
        else:
            return None
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def get_zestimate_from_address(driver, address):
    """
    Get an estimated rent for a given address by searching Zillow listings on Google and processing the closest match.

    :param driver: A Selenium webdriver object
    :param address: A string representing the address.

    :return: An integer value representing the estimated rent.
    """
    # Convert address to a Google search URL
    search_query = f'intitle:"{address}" zillow'
    encoded_query = urllib.parse.quote(search_query)
    google_search_url = f'https://www.google.com/search?q={encoded_query}'

    zillow_narrow_pattern = r'zillow\.com/homedetails/([^\s//?]+)'
    zillow_must_contain = ["zillow"]
    must_not_contain = ["apartments.com", "trulia.com", "loopnet.com", "realtor.com"]
    zillow_prefix = "https://www.zillow.com/homedetails/"

    task_data = {
        'driver': driver,
        'search_url': google_search_url,
        'path_ext_pattern': zillow_narrow_pattern,
        'must_contain': zillow_must_contain,
        'must_not_contain': must_not_contain,
        'prefix': zillow_prefix,
    }

    # Get a list of cached URLs
    cached_urls = g_scraper.get_google_cached_urls_from_google_search_urls(task_data)

    # Process and compare each URL to find the closest match
    closest_match = None
    for url in cached_urls:
        # Extract and clean the address part from the URL
        extracted_address = re.sub(r'[^a-zA-Z0-9\s]', ' ', url.split('/')[5]).strip()

        if extracted_address == address.replace(',', '').replace('-', ' '):
            print("Exact match")
            closest_match = url
            break

        # If no exact match, continue to find the closest match using TFIDF
        # This part will be implemented below

    # Use the closest match URL to get the estimated rent
    estimated_rent = get_zestimate_from_zillow_google_cache_url(closest_match)
    return estimated_rent


def analyze_reviews_sentiments(reviews):
    """
    Analyzes the sentiments of a list of reviews and returns the most positive and most negative reviews.

    :param reviews: A list of reviews, where each review is a dictionary with 'text' and 'date' keys.
    :return: A dictionary with two keys 'most_positive' and 'most_negative' each containing the top 2 reviews.
    """
    try:
        # Download VADER lexicon for sentiment analysis
        nltk.download('vader_lexicon')

        # Initialize the Sentiment Intensity Analyzer
        sia = SentimentIntensityAnalyzer()

        # Analyze sentiment of each review
        sentiments = [
            {'text': review['text'], 'date': review['date'], 'score': sia.polarity_scores(review['text'])['compound']}
            for review in reviews]

        # Sort the reviews by sentiment score
        sorted_sentiments = sorted(sentiments, key=lambda x: x['score'])

        # Get 2 most positive and 2 most negative reviews
        most_positive = sorted_sentiments[-2:]
        most_negative = sorted_sentiments[:2]

        return {'most_positive': most_positive, 'most_negative': most_negative}
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return {'most_positive': None, 'most_negative': None}


def clean_text(text):
    """
    Cleans the given text by removing non-English characters and converting it to lowercase.

    This function removes any characters that are not ASCII (non-English) and then removes any 
    characters that are not letters. The resulting text is converted to lowercase to standardize it.

    :param text: The text string to be cleaned.

    :return: The cleaned and lowercased text string.

    Called by: clean_and_process_menu_data
    """
    try:
        text = re.sub(r'[^\x00-\x7F]+', ' ', text).lower()
        text = re.sub('[^A-Za-z ]+', '', text).lower()
        return text
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return ""


def price_to_float(price_str):
    """
    Converts a price string to a floating-point number.

    This function removes any non-numeric characters from the price string, including currency symbols,
    and then converts the remaining string to a float.

    :param price_str: The price string to convert.

    :return: The price as a floating-point number.

    Called by: clean_and_process_menu_data
    """
    try:
        return float(re.sub(r'[^\d.]', '', price_str))
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return 0


def clean_and_process_menu_data(menu_data):
    """
    Processes a list of menu items by cleaning item names and converting prices to floats.

    For each item in the menu data, this function cleans the item name using `clean_text` and 
    converts the item price to a float using `price_to_float`.

    :param menu_data: A list of dictionaries, each representing a menu item with 'item_name' and 'item_price' keys.

    :return: The processed menu data with cleaned item names and float prices.

    Called by: (Typically called as part of a larger data processing workflow)
    Calls: clean_text, price_to_float
    """
    try:
        for item in menu_data:
            item['item_name'] = clean_text(item['item_name'])
            item['item_price'] = price_to_float(item['item_price'])
        return menu_data
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def calculate_common_words(menu_data, min_word_length=3):
    """
    Calculates the most common words in the menu data, excluding short and common (stop) words.

    This function compiles all the item names from the menu data into a single text, 
    then splits this text into words. It filters out common English stop words and words 
    shorter than a specified minimum length. The function then counts the occurrences of 
    each word and returns the most common ones.

    :param menu_data: A list of dictionaries, each representing a menu item with an 'item_name' key.
    :param min_word_length: The minimum length of words to include in the count.

    :return: A list of tuples representing the most common words and their counts.

    Called by: calculate_average_prices
    """
    try:
        nltk_stopwords = set(stopwords.words('english'))
        all_words = ' '.join(item['item_name'] for item in menu_data).split()
        filtered_words = [word for word in all_words if
                          word not in nltk_stopwords and len(word) >= min_word_length and word.isalpha()]
        return Counter(filtered_words).most_common(5)
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return Counter([0])


def calculate_average_prices(menu_data, common_words):
    """
    Calculates average prices for items associated with common words and overall average prices.

    This function analyzes the menu data to compute average prices for items related to each of 
    the common words. It also calculates the average price of bestseller items and the overall 
    average price of all items.

    :param menu_data: A list of dictionaries, each representing a menu item with 'item_name' and 'item_price' keys.
    :param common_words: A list of common words to calculate average prices for.

    :return: A dictionary with average prices for each common word and overall averages.

    Called by: (Typically called as part of a larger data processing or analysis workflow)
    """
    try:
        word_prices = {word: [] for word, _ in common_words}
        bestseller_prices, all_prices = [], []

        for item in menu_data:
            price = item['item_price']
            all_prices.append(price)
            if item['item_is_bestseller']:
                bestseller_prices.append(price)
            for word in item['item_name'].split():
                if word in word_prices:
                    word_prices[word].append(price)

        average_bestseller_price = sum(bestseller_prices) / len(bestseller_prices) if bestseller_prices else 0
        average_price = sum(all_prices) / len(all_prices) if all_prices else 0
        average_prices_common_items = {word: sum(prices) / len(prices) for word, prices in word_prices.items()}

        return average_bestseller_price, len(menu_data), average_price, average_prices_common_items
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return None, None, None, None

# Code example
# Process data
# menu_data = clean_and_process_menu_data(menu_data)
# common_words = calculate_common_words(menu_data)
# average_bestseller_price, num_items, overall_average_price,
#       average_prices_common_items = calculate_average_prices(menu_data, common_words)

# # Output results
# print("Average Bestseller Price:", average_bestseller_price)
# print("Number of Items:", num_items)
# print("Overall Average Price:", overall_average_price)
# print("Average Prices for Common Words:", average_prices_common_items)


def convert_date(date_str):
    """
    Converts a date string to a standard 'YYYY-MM-DD' format.

    This function handles both relative dates (like "1 day ago") and specific dates (like "Nov 13, 2023").
    It calculates the absolute date for relative dates based on the current date.

    :param date_str: The date string to be converted.

    :return: A string representing the date in 'YYYY-MM-DD' format.
    """
    # Handle relative dates (e.g., "1 day ago", "2 weeks ago")
    try:
        date = None
        if "ago" in date_str:
            num, unit = date_str.split()[:2]
            num = int(num)
            if "day" in unit:
                date = datetime.now() - timedelta(days=num)
            elif "week" in unit:
                date = datetime.now() - timedelta(weeks=num)
            return date.strftime('%Y-%m-%d')

        # Handle specific dates (e.g., "Nov 13, 2023")
        else:
            date = datetime.strptime(date_str, '%b %d, %Y')
            return date.strftime('%Y-%m-%d')
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def calculate_avg_days_between_reviews(reviews):
    """
    Calculates the average number of days between consecutive reviews.

    This function parses the dates of reviews, sorts them, and calculates the time
    difference between consecutive reviews. It then computes the average of these differences.

    :param reviews: A list of review dictionaries, each containing a 'date' key with the review date.

    :return: The average number of days between consecutive reviews.
    """
    try:
        dates = [datetime.strptime(review['date'], "%Y-%m-%d") for review in reviews]
        dates.sort()
        days_between_reviews = [dates[i + 1] - dates[i] for i in range(len(dates) - 1)]
        avg_days_between_reviews = np.mean([daysbetween.days for daysbetween in days_between_reviews])
        return avg_days_between_reviews
    except Exception as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return "10000000"
