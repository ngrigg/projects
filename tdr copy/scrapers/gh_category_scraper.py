from utils import db_manager as ngdb
from scrapers import scraper_utils as s_utils
from utils import misc_utils as m_utils
from celery_app import app
import traceback
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException
from datetime import datetime
import re
CUISINES_FILE = 'grub_cuisines.txt'


@app.task(name="scraper_app.get_gh_restaurant_links_from_google_cache_gh_category_pages")
def get_gh_restaurant_links_from_google_cache_gh_category_pages(task_data):
    """
    Scrapes Grubhub restaurant URLs from a specific Google cache category page.

    This function is called by a task scheduler as part of a larger scraping workflow. 
    It receives a WebDriver instance and other necessary data through the task_data argument, 
    processes the specified Google cache URL, and extracts Grubhub restaurant URLs.

    Parameters:
    - task_data (dict): A dictionary containing the necessary data for scraping, including the WebDriver 
      instance, the URL to scrape, and other relevant data.
    """
    driver = task_data['driver']
    session_id, executor_url = driver['session_id'], driver['executor_url']
    driver = s_utils.create_driver_session(session_id, executor_url)
    google_cache_url = task_data['search_url']
    task = task_data['task']

    try:
        # Extracting necessary variables from task_data
        print("Starting get_gh_restaurant_links_from_google_cache_gh_category_pages")
        print(f"Processing URL: {google_cache_url}")

        # Logging process initiation and updating task status
        ngdb.insert_one('logs', {'message': f"Started processing: {google_cache_url}", 'timestamp': datetime.now()})
        ngdb.upsert_one('tasks', {'url': google_cache_url}, {'url': google_cache_url, 'status': 'pending'})

        if "grubhub" not in google_cache_url or "google" not in google_cache_url:
            print(f"Invalid URL. Skipped scraping. URL: {google_cache_url}")
            # Update task status to indicate the URL is skipped
            ngdb.update_one('tasks', {'url': google_cache_url}, {'$set': {'status': 'skipped'}})
            return

        print("Determining scraper type...")
        scraper_type = determine_scraper_type(driver, google_cache_url, CUISINES_FILE)
        print(f"Scraper type is {scraper_type}")

        restaurant_data = process_grubhub_directory_page_helper(driver, google_cache_url, scraper_type)

        # Upsert the unique URLs into the database
        print("Now attempting to enter new task information into database...")
        new_tasks = []
        for data in restaurant_data:
            new_task = {
                'params': {"url": data['url']}, 
                'last_step': "get_gh_restaurant_links_from_google_cache_gh_category_pages",
                'next_step': "",
                'scrape_attempted': False,
            }
            print("added_new_task to database for persistence across sessions - remember to add to queue", new_task)
            new_tasks.append(new_task)
        result = ngdb.upsert_many("tasks", new_tasks, "params.url")
        if result:
            print(f"Document inserted with ID: {result}")
        else:
            print(f"Document already exists.")

        # Task completion is handled by execute_task function in scheduler.py

        # print("task: ", task)
        #
        # # scheduler.handle_task_completion(task, 'completed')
        # print(f"Completed processing URL: {google_cache_url}")

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        m_utils.take_screenshot(driver)
        print(f"Full traceback: {traceback.format_exc()}")
        # ngdb.log_error_to_database(db, str(e))

        # if task_data.get('retry_count', 0) < MAX_RETRY_LIMIT:
        #     task_data['retry_count'] = task_data.get('retry_count', 0) + 1
        #     scheduler.enqueue_task(task_data)
        #     print(f"Re-enqueued task with incremented retry count: {task_data['retry_count']}")


def process_grubhub_directory_page_helper(driver, url, scraper_type):
    """
    Helper function to provide the correct inputs to process_grubhub_directory_page
    based on the scraper type (cuisine or not).

    Parameters:
    - driver: WebDriver instance
    - url: URL to be processed
    - scraper_type: Type of scraper ('cuisine' or other)
    - various selectors (excluded for brevity)
    """

    cuisine_restaurant_card_chain = [
        ["div.searchResults-items-container > div > span > span > span > div > div"],
        ["div.s-row > div.s-col-xs-12.s-col-md-9 > div > div > span > span:nth-child(4) > span > div > div"],
        ["div[data-testid='restaurant-card']"]
    ]
    cuisine_url_chain = [
        ["a.restaurant-name.u-text-wrap.s-card-title--darkLink"],
        ["a[data-testid='restaurant-name']"],
        ["div[data-testid='restaurant-card'] > a.restaurant-name"]
    ]
    cuisine_name_chain = [
        # ["h5[data-testid='restaurant-name']"],
        ["a[data-testid='restaurant-name'] > h5"],
        ["h5.u-text-ellipsis"],
        ["h5"]
    ]
    cuisine_category_chain = [
        ["span[data-testid='cuisines'] > span"],
        ["div.restaurantCard-primaryInfo > div > div > div > span:nth-child(2) > span"],
        ["div.restaurantCard-detailsContainer > div.restaurantCard-primaryInfo > div > div > div > span:nth-child(2) > span"]
    ]
    cuisine_review_count_chain = [
        ["span[data-testid='star-rating-text']"],
        ["div.restaurantCard-tertiaryInfo > div.restaurantCard-rating > span > div > span > div > span.sc-dkrFOg.cbuoQk"],
        ["div.restaurantCard-tertiaryInfo-wrapper > div.restaurantCard-rating > span > div > span > div > span"]
    ]
    neighborhood_restaurant_card_chain = [
        ["div[data-testid='restaurant-card']"],
        ["div.s-card.s-card-wrapper.s-card-wrapper--fullCTA.u-block[data-testid='card-template']"],
        ["div.u-gutter-2.s-col-sm-6.s-col-md-4.s-col-lg-3.u-stack-y-4"]
    ]
    neighborhood_url_chain = [["a[data-testid='restaurant-name']"], 
                              ["a.restaurant-name.u-text-wrap.s-card-title--darkLink"]]
    neighborhood_name_chain = [["h5[data-testid='restaurant-name']"], 
                               ["a[data-testid='restaurant-name'] > h5"]]
    neighborhood_category_chain = [["div.s-card-body", "span:nth-child(2)"], 
                                   ["span[data-testid='cuisines'] > span"]]
    neighborhood_review_count_chain = [["div.content-ratings", "span[data-testid='star-rating-text']"], 
                                       ["div[data-testid='content-ratings'] > span > div > span[data-testid='star-rating-text']"]]  

    if scraper_type == "cuisine":
        print("Running process_grubhub_directory_page scraper for GH CUISINE DIRECTORY url of format: /delivery/NEIGHBORHOOD/CUISINE")
        return process_grubhub_directory_page(driver, url, cuisine_restaurant_card_chain, cuisine_url_chain, cuisine_name_chain, cuisine_category_chain, cuisine_review_count_chain, max_errors=5)
    else:
        print("Running process_grubhub_directory_page scraper for GH NEIGHBORHOOD DIRECTORY url of format: /delivery/NEIGHBORHOOD")
        return process_grubhub_directory_page(driver, url, neighborhood_restaurant_card_chain, neighborhood_url_chain, neighborhood_name_chain, neighborhood_category_chain, neighborhood_review_count_chain, max_errors=5)


def process_grubhub_directory_page(driver, google_cache_url, restaurant_card_chain, url_chain, name_chain, category_chain, review_count_chain, max_errors):

    """
    Processes a Grubhub directory page to extract restaurant data.

    This function navigates to a Grubhub directory page and extracts data for each restaurant card
    on the page. The data includes the restaurant's URL, name, category, and review count. It handles
    errors up to a specified threshold.

    :param driver: The WebDriver instance used for the webpage.
    :param google_cache_url: The URL of the Grubhub directory page.
    :param max_errors: The maximum number of allowed errors before stopping the process.
    :param restaurant_card_chain: The selector chains to locate restaurant cards.
    :param url_chain: The selector chains to extract the restaurant URL.
    :param name_chain: The selector chains to extract the restaurant name.
    :param category_chain: The selector chains to extract the restaurant category.
    :param review_count_chain: The selector chains to extract the restaurant review count.

    :return: A list of dictionaries, each containing data for a restaurant.

    Called by: scrape_grubhub_google_cache_urls
    Calls: find_restaurant_cards, get_element_text_or_default_chain, extract_page_number
    """    
    print("Running process_grubhub_directory_page to gather GH restaurant URLs from GH directory page...")
    error_count = 0
    page_number = extract_page_number(google_cache_url)
    restaurant_data = []
    card_position = 0

    try:
        # Wait for the restaurant cards to be present
        WebDriverWait(driver, 15).until(
            lambda d: any_of_selectors_present(d, restaurant_card_chain)
        )
        m_utils.take_screenshot(driver, google_cache_url)

        # Find restaurant cards
        restaurant_cards = find_restaurant_cards(driver, restaurant_card_chain)
        if not restaurant_cards:
            return []

        print(f"Found {len(restaurant_cards)} number of restaurant cards on page.")

        # Process each restaurant card
        for card in restaurant_cards:
            try:
                card_position += 1
                overall_rank = card_position + (page_number - 1) * 36  # Assuming 36 cards per page

                # Extracting data using passed selector chains
                url = s_utils.get_element_text_or_default_chain(card, url_chain, default="URL not found", attribute='href')
                name = s_utils.get_element_text_or_default_chain(card, name_chain, default="Name not found")
                category = s_utils.get_element_text_or_default_chain(card, category_chain, default="Category not found")
                num_reviews = s_utils.get_element_text_or_default_chain(card, review_count_chain, default="Review count not found")

                if not name or not url:
                    error_count += 1
                    if error_count > max_errors:
                        print(f"Exceeded error threshold. Processed {len(restaurant_data)} restaurants.")
                        break

                if num_reviews and "ratings" in num_reviews:
                    try:
                        num_reviews = int(num_reviews.split()[0])
                    except ValueError:
                        print("Issue processing number of reviews")
                        num_reviews = "Review count not found"
                else:
                    print("Issue processing number of reviews")
                    num_reviews = "Review count not found"
                
                new_data = {
                    'name': name,
                    'url': url,
                    'num_reviews': num_reviews,
                    'category': category,
                    'impression_rank': overall_rank
                }

                restaurant_data.append(new_data)
                print("Category scraper gathered new data", new_data)
                
            except (TimeoutException, NoSuchElementException, WebDriverException) as e:
                print("Exception", e)
                print(f"Full traceback: {traceback.format_exc()}")
                error_count += 1
                if error_count > max_errors:
                    print("Exceeded maximum error threshold, continuing with what has been processed.")
                    break
                continue

        return restaurant_data

    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        m_utils.take_screenshot(driver, google_cache_url)
        return []


def determine_scraper_type(driver, url, cuisines):
    """
    Determines the type of scraper to use based on the URL and the page title.

    This function analyzes the given URL and the page title to decide whether the scraper should
    target a specific cuisine or a neighborhood. It resolves any conflicts between the URL and title
    determinations.

    :param driver: The WebDriver instance used for the webpage.
    :param url: The URL of the page.
    :param cuisines: A set of cuisines to check against.

    :return: The determined scraper type ('cuisine' or 'neighborhood').

    Called by: scrape_grubhub_google_cache_urls
    Calls: get_element_text_or_default_chain
    """    
    url = url.lower()
    
    # Extract possible cuisine from URL
    match = re.search(r'grubhub.com/delivery/[^/]+/([^/%]+)', url)
    url_scraper_type = "neighborhood"
    if match:
        extracted_text = match.group(1)
        for cuisine in cuisines:
            if cuisine in extracted_text:
                url_scraper_type = "cuisine"
                break
    
    # Selector chain for the title element
    primary_selector = "h1[data-testid='hero-title']"
    secondary_selector = "h1.massive-heading"
    selector_chain = [primary_selector, secondary_selector]

    # Extract page title using get_element_text_or_default_chain
    page_title = s_utils.get_element_text_or_default_chain(driver, selector_chain, default=None)

    if page_title:
        # Process page title and determine scraper type
        title_scraper_type = "neighborhood" if page_title.lower().split()[0] == "food" else "cuisine"
        print(f"Page title: {page_title}, Cuisine match: {match.group(1) if match else 'None'}")
    else:
        # Default to URL determination if title extraction fails
        title_scraper_type = url_scraper_type

    # Conflict resolution
    if url_scraper_type != title_scraper_type:
        print("Warning: titles do not match.")
        # Default to title_scraper_type in case of conflict
        return title_scraper_type

    return url_scraper_type


def any_of_selectors_present(driver, selector_chains):
    """
    Checks if any of the provided selector chains match an element on the webpage.

    This function iterates through a list of selector chains and checks if any element matches
    the selectors in the chain. It returns True if any element is found, otherwise False.

    :param driver: The WebDriver instance used for the webpage.
    :param selector_chains: A list of selector chains to try.

    :return: True if any element is found, otherwise False.

    Called by: process_grubhub_directory_page
    """    
    for selectors_chain in selector_chains:
        try:
            if driver.find_element(By.CSS_SELECTOR, " > ".join(selectors_chain)):
                return True
        except (TimeoutException, NoSuchElementException, WebDriverException) as e:
            print("Exception", e)
            print(f"Full traceback: {traceback.format_exc()}")
            continue
    return False


def extract_page_number(url):
    """
    Extracts the page number from a URL string.

    This function searches for a 'pageNum' parameter in the URL and extracts its numeric value.
    If no such parameter is found, it defaults to returning page number 1.

    :param url: The URL string to extract the page number from.

    :return: The extracted page number as an integer.

    Called by: process_grubhub_directory_page
    """    
    match = re.search(r'pageNum=(\d+)', url)
    return int(match.group(1)) if match else 1


def find_restaurant_cards(driver, selector_chains):
    """
    Finds and returns restaurant card elements from a webpage using multiple selector chains.

    This function iterates through a list of selector chains, trying each chain to locate restaurant
    card elements on the page. If a set of elements is found using any chain, they are returned.

    :param driver: The WebDriver instance used for the webpage.
    :param selector_chains: A list of selector chains to try for finding the restaurant cards.

    :return: A list of WebElement objects representing the restaurant cards, or an empty list if none found.

    Called by: process_grubhub_directory_page
    """    
    for selectors_chain in selector_chains:
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, " > ".join(selectors_chain))
            if cards:
                return cards
        except NoSuchElementException as e:
            print("Exception", e)
            print(f"Full traceback: {traceback.format_exc()}")
            continue
    return []
