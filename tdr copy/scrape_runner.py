# /scrape_runner.py

from utils import db_manager as ngdb, misc_utils as m_utils
from task_mgmt import tasks
import itertools
import logging
import urllib.parse
LARGE_CHAINS_FILE = 'large_chains.txt'
NEIGHBORHOODS_FILE = 'grub_neighborhoods.csv'
CUISINES_FILE = 'grub_cuisines.txt'


def get_starting_google_search_urls(neighborhoods_file, cuisines_file, large_chains_file):
    """
    Generates search URLs for Grubhub restaurants and dispatches them as tasks to Celery.

    Parameters:
    - neighborhoods_file (str): File path to CSV containing neighborhood data.
    - cuisines_file (str): File path to text file containing list of cuisines.
    - large_chains_file (str): File path to text file containing list of large chains.
    """
    print("Generating and storing search URLs")
    neighborhoods = m_utils.read_csv(neighborhoods_file)
    cuisines = m_utils.read_txt(cuisines_file)
    chains = m_utils.read_txt(large_chains_file)

    new_tasks = []
    combined_items = itertools.chain(itertools.product(neighborhoods, cuisines),
                                     itertools.product(neighborhoods, chains))

    for (borough, neighborhood), item in combined_items:
        quoted_neighborhood = urllib.parse.quote_plus(neighborhood)
        quoted_item = urllib.parse.quote_plus(item)

        for page in range(1, 4):
            search_url = f"https://www.google.com/search?q=https%3A%2F%2Fwww.grubhub.com%2F+{borough}+{quoted_neighborhood}+{quoted_item}+%22page+{page}%22"
            new_task = {
                'params': {"search_url": search_url},
                'last_step': "get_starting_google_search_urls",
                'next_step': "get_google_cached_urls_from_google_search_urls",
                'scrape_attempted': False,
            }
            new_tasks.append(new_task)

    ngdb.insert_many("tasks", new_tasks)
    return


def test_gh_category_scraper(test_cases):
    """
    Test the Grubhub category scraper with specified URLs.

    This function tests the Grubhub category scraper by dispatching tasks to the Celery queue.
    It first deletes any existing tasks for the category scraper in the database and then
    creates new tasks based on the provided test URLs.

    Parameters:
    - test_cases (list): A list of URLs to test the scraper with.

    The function interacts with the database to manage the test tasks and uses the Celery
    system to dispatch these tasks for processing.
    """
    print("Attempting to delete existing tasks.")
    # Deleting existing test tasks related to category scraper
    ngdb.bulk_delete('tasks', [{'next_step': 'get_gh_restaurant_links_from_google_cache_gh_category_pages'}])
    print("Existing test tasks for category scraper deleted.")

    # Creating and dispatching new test tasks for each test case
    new_tasks = []
    for test_case in test_cases:
        new_task = {
            'params': {'search_url': test_case},
            'last_step': 'get_google_cached_urls_from_google_search_urls',
            'next_step': 'get_gh_restaurant_links_from_google_cache_gh_category_pages',
            'scrape_attempted': False,
        }
        new_tasks.append(new_task)
    ngdb.insert_many("tasks", new_tasks)


def test_gh_restaurant_scraper(test_cases):
    """
    Test the Grubhub restaurant data scraper with specified URLs.

    This function tests the Grubhub restaurant data scraper by dispatching tasks to the Celery queue.
    It first deletes any existing tasks for the restaurant data scraper in the database and then
    creates new tasks based on the provided test URLs.

    Parameters:
    - test_cases (list): A list of URLs to test the scraper with.

    The function interacts with the database to manage the test tasks and uses the Celery
    system to dispatch these tasks for processing.
    """

    # Deleting existing test tasks related to restaurant data scraper
    ngdb.bulk_delete('tasks', [{'next_step': 'get_gh_restaurant_data_from_google_cache_gh_restaurant_page'}])
    print("Existing test tasks for restaurant data scraper deleted.")

    # Creating and dispatching new test tasks for each test case
    new_tasks = []
    for test_case in test_cases:
        new_task = {
            'params': {'search_url': test_case},
            'last_step': 'get_google_cached_urls_from_google_search_urls',
            'next_step': 'get_gh_restaurant_data_from_google_cache_gh_restaurant_page',
            'scrape_attempted': False,
        }
        new_tasks.append(new_task)
    ngdb.insert_many("tasks", new_tasks)


def main():
    """
    The main function to initiate the scraping process.
    This function initializes the database, generates search URLs,
    and dispatches tasks to the Celery queue.
    """
    # # Generating search URLs for Grubhub restaurants
    # print("Generating search URLs")
    # get_starting_google_search_urls(NEIGHBORHOODS_FILE, CUISINES_FILE, LARGE_CHAINS_FILE)

    # # Test cases for category and restaurant scrapers
    # restaurant_scraper_test_cases = [
    #     'https://webcache.googleusercontent.com/search?q=cache:2QUQTsv3SogJ:https://www.grubhub.com/restaurant/qq-house-65th-st-2229-65th-st-brooklyn/2760987/reviews&hl=en&gl=us',
    #     'https://webcache.googleusercontent.com/search?q=cache:pkJ3MZIFS_wJ:https://www.grubhub.com/restaurant/taro-sushi-244-flatbush-ave-brooklyn/66754&hl=en&gl=us'
    # ]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    m_utils.check_redis_connection()
    ngdb.debug_database_connection('mongodb', 27017, 'init_user', 'init_pwd',
                                   'tdr_case', 'tasks')
    # m_utils.check_and_manage_docker_redis() # Not meaningful test when you are on docker

    category_scraper_test_cases = [
        'https://webcache.googleusercontent.com/search?q=cache:2qYEckZ7NMMJ:https://www.grubhub.com/delivery/ny-nyc/mexican&hl=en&gl=us',
        'https://webcache.googleusercontent.com/search?q=cache:qbg_xVXLg5EJ:https://www.grubhub.com/delivery/ny-nyc/chinese&hl=en&gl=us'
    ]

    # Dispatching test tasks for category and restaurant scrapers
    print("Dispatching test tasks for category scraper")
    # test_gh_category_scraper(category_scraper_test_cases)

    # print("Dispatching test tasks for restaurant scraper")
    # test_gh_restaurant_scraper(restaurant_scraper_test_cases)

    # Dispatching tasks for the main scraping process
    print("Dispatching tasks for scraping process")
    # tasks.dispatch_tasks.apply_async(args=[{'scrape_attempted': False, 'next_step': 'get_gh_restaurant_links_from_google_cache_gh_category_pages'}], kwargs={'task_load_limit': 5})

    print("scrape_runner.py ran successfully.")


if __name__ == "__main__":
    main()
