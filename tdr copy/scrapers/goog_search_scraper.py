from utils import db_manager as ngdb
from scrapers import scraper_utils as s_utils
from celery_app import app
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
import re
from datetime import datetime
import traceback

# Global variables
MAX_RETRY_LIMIT = 4
WEBDRIVER_INIT_RETRIES = 3
MIN_LINKS_LOADED = 1
LARGE_CHAINS_FILE = 'large_chains.txt'
NEIGHBORHOODS_FILE = 'grub_neighborhoods.csv'
CUISINES_FILE = 'grub_cuisines.txt'
SCRAPEOPS_API_KEY = 'b77ca529-e955-499d-8183-b258251ee3c4'


WAIT_CONDITIONS_DEFAULT = [
    {'condition': lambda d: d.execute_script('return document.readyState') == 'complete',
     'timeout': 25, 'description': "Page Load Complete"},
    {'condition': ec.presence_of_element_located((By.TAG_NAME, 'body')),
     'timeout': 25, 'description': "Body Tag Loaded"},
    # {'condition': links_loaded,
    #  'timeout': 15, 'description': "At Least 3 Search Result Links Loaded"}
]

VALIDITY_CHECKS_DEFAULT = [
        {'condition': lambda d: d.find_element(By.TAG_NAME, "body").get_attribute('innerHTML').strip() != "",
         'message': "Empty body content"},
        {'condition': lambda d: all(indicator not in d.title.lower() for indicator in ["error", "not found", "403 forbidden", "access denied"]),
         'message': "Error indicator found in title"}
    ]


@app.task(name="scraper_app.get_google_cached_urls_from_google_search_urls")
def get_google_cached_urls_from_google_search_urls(task_data):
    """
    Extracts and constructs possible Google cache URLs from a given search URL.

    Parameters:
    - task_data (dict): Contains necessary data for scraping including driver, search_url, etc.
    - db: The database connection object.
    - scheduler: The task scheduler instance.

    Returns:
    A list of possible Google cache URLs that match the given criteria.
    """
    print("starting get_google_cached_urls_from_google_search_urls")
    driver = task_data['driver']
    session_id, executor_url = driver['session_id'], driver['executor_url']
    driver = s_utils.create_driver_session(session_id, executor_url)
    url = task_data['search_url']
    path_ext_pattern = task_data['path_ext_pattern']
    must_contain = task_data['must_contain']
    must_not_contain = task_data['must_not_contain']
    prefix = task_data['prefix']
    last_step = task_data['last_step']
    print("task_data", task_data)

    try:
        # Log process start and mark task pending
        ngdb.insert_one('logs',
                        {'message': f"Started processing: {task_data['search_url']}", 'timestamp': datetime.now()})
        ngdb.upsert_one('tasks', {'url': url}, {'url': url, 'status': 'pending'})

        page_content = driver.page_source
        cache_pattern_wide_net = r'webcache.{0,500}'
        wide_matches = re.findall(cache_pattern_wide_net, page_content)
        must_contain += ['google']

        cache_pattern_uid = r'cache:([A-Za-z0-9]+):h'

        possible_urls = []
        for wide_match in wide_matches:
            cache_uid_match = re.search(cache_pattern_uid, wide_match)
            path_ext_match = re.search(path_ext_pattern, wide_match)

            if cache_uid_match and path_ext_match:
                cache_uid = cache_uid_match.group(1)
                path_extension = path_ext_match.group(1)
                if len(path_ext_match.groups()) >= 2 and path_ext_match.group(2):
                    page_num = path_ext_match.group(2) if path_ext_match.group(2) else ''
                else:
                    page_num = ''
                page_num_extension = f"?pageNum={page_num}" if page_num else ''
                google_cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{cache_uid}:{prefix}{path_extension}{page_num_extension}&hl=en&gl=us"
                if google_cache_url.endswith('-') or google_cache_url.endswith('_'):
                    print(f"Excluded URL due to invalid ending: {google_cache_url}")
                else:
                    if all(keyword in google_cache_url for keyword in must_contain) and not any(
                            banned in google_cache_url for banned in must_not_contain):
                        possible_urls.append(google_cache_url)

        # Remove duplicate URLs
        unique_urls = list(set(possible_urls))

        # Upsert the unique URLs into the database
        new_tasks = []
        for url in unique_urls:
            if last_step == "get_starting_google_search_urls":
                new_task = {
                    'params': {"url": url},
                    'last_step': "get_google_cached_urls_from_google_search_urls",
                    'next_step': "",
                    'scrape_attempted': False,
                }
            elif last_step == "get_google_cached_urls_from_google_search_urls":
                new_task = {}
            else:
                new_task = {}
                print("******************************")
                print("Warning: Error with last step.")
                print("******************************")

            print("added_new_task to database for persistence across sessions - remember to add to queue", new_task)
            new_tasks.append(new_task)
            result = ngdb.insert_many("tasks", new_tasks)
            if result:
                print(f"Document inserted with ID: {result}")
            else:
                print(f"Document already exists: {result['url']}")
        return

    except Exception as e:
        print("General exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        ngdb.log_error_to_database(str(e))
        return
