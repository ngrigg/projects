#!/usr/bin/env python
# coding: utf-8
from celery_app import app
from utils import db_manager as ngdb
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
import time
import requests
import psutil
import json
from random import choice
import random
import threading
import traceback
import subprocess
from utils import misc_utils

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
]

VALIDITY_CHECKS_DEFAULT = [
        {'condition': lambda d: d.find_element(By.TAG_NAME, "body").get_attribute('innerHTML').strip() != "", 
         'message': "Empty body content"},
        {'condition': lambda d: all(indicator not in d.title.lower() for indicator in ["error", "not found", "403 forbidden", "access denied"]), 
         'message': "Error indicator found in title"}
    ]


@app.task(name="scraper_app.fetch_and_process_page_helper")
def fetch_and_process_page_helper(task_data):
    """
    A helper function to adapt the fetch_and_process_page task for the task chain.

    :param task_data: The dictionary containing task arguments.
    :return: Modified task_data including the results from fetch_and_process_page.
    """
    # Extracting required parameters from task_data
    driver = task_data['prelim_args'].get('driver')
    url = task_data['prelim_args'].get('url')
    proxy_performance_threshold = task_data['prelim_args'].get('proxy_performance_threshold', 30)
    max_retries = task_data['prelim_args'].get('max_retries', 4)
    wait_conditions = task_data['prelim_args'].get('wait_conditions', WAIT_CONDITIONS_DEFAULT)
    validity_checks = task_data['prelim_args'].get('validity_checks', VALIDITY_CHECKS_DEFAULT)
    return_driver = task_data['prelim_args'].get('return_driver', False)

    # Call the actual fetch_and_process_page function
    result = fetch_and_process_page(driver, url, proxy_performance_threshold, max_retries,
                                    wait_conditions, validity_checks, return_driver)

    # Update task_data with the results from fetch_and_process_page
    task_data['args']['driver'] = result['driver']

    return task_data


def fetch_and_process_page(driver, url, proxy_performance_threshold=30, max_retries=4, wait_conditions=WAIT_CONDITIONS_DEFAULT, validity_checks=VALIDITY_CHECKS_DEFAULT, return_driver=False):
    attempt = 0
    last_request_timestamps = []  # To track request timestamps for rate limiting

    def load_url(local_driver, local_url):
        try:
            local_driver.get(local_url)
        except (TimeoutException, NoSuchElementException, WebDriverException) as e1:
            print(f"Error loading URL: {e1}")
            print(f"Full traceback: {traceback.format_exc()}")
            ngdb.log_error_to_database(f"Error loading URL: {e1}")

    rate_limit = 10  # Max number of requests
    time_window = 60  # Time window in seconds

    while attempt < max_retries:
        try:
            # Rate Limiting Check
            current_time = time.time()
            last_request_timestamps = [t for t in last_request_timestamps if current_time - t < time_window]
            if len(last_request_timestamps) >= rate_limit:
                print("Rate limit exceeded. Sleeping.")
                time.sleep(time_window - (current_time - last_request_timestamps[0]))
            last_request_timestamps.append(current_time)

            # Test proxy performance
            proxy_performance = test_proxy_performance(driver)
            if proxy_performance == 0 or proxy_performance > proxy_performance_threshold:
                print("Poor proxy load time. Reinitializing WebDriver.")
                driver.quit()
                driver = initialize_selenium()

            print("Proxy performance seems OK.")
            print(f"Attempting to load URL (Attempt: {attempt + 1}): {url}")

            # Start driver.get(url) in a separate thread
            thread = threading.Thread(target=load_url, args=(driver, url))
            thread.start()
            thread.join(timeout=45)  # Set timeout as needed

            if thread.is_alive():
                print(f"Loading URL timed out. Retrying...")
                misc_utils.take_screenshot(driver, url)
                ngdb.log_error_to_database("Loading URL timed out")
                attempt += 1
                continue

            print("driver.get(url) completed")
            print("Evaluating wait conditions...")

            if wait_conditions:
                for condition in wait_conditions:
                    try:
                        print(f"Waiting for condition: {condition['description']}")
                        WebDriverWait(driver, condition['timeout']).until(condition['condition'])
                    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
                        print(f"Exception: {e}")
                        print(f"Full traceback: {traceback.format_exc()}")
                        ngdb.log_error_to_database(f"Timeout while waiting for condition: {condition['description']}")
                        attempt += 1
                        continue  # Skip to next iteration of the loop

            print("Getting page source from driver.")
            page_content = driver.page_source

            print("Checking validity of page.")
            if not check_page_is_valid(page_content, driver, validity_checks):
                print("Page is not valid or blocked, retrying...")
                misc_utils.take_screenshot(driver, url)
                driver = initialize_selenium()
                attempt += 1
                continue

            session_id, executor_url = save_driver_session(driver)

            print("Page seems valid. Returning results.")
            if return_driver:
                return {"driver": {"session_id": session_id, "executor_url": executor_url}, "page_source": page_content}
            else:
                return {"page_source": page_content}

        except (TimeoutException, NoSuchElementException, WebDriverException) as e:
            print(f"Exception encountered: {e}. Retrying...")
            print(f"Full traceback: {traceback.format_exc()}")
            misc_utils.take_screenshot(driver, url)
            ngdb.log_error_to_database(f"Exception encountered: {e}")
            if driver:
                driver.quit()
            attempt += 1
            continue

        except Exception as e:
            print(f"General exception encountered: {e}. Taking a screenshot and aborting.")
            print(f"Full traceback: {traceback.format_exc()}")
            misc_utils.take_screenshot(driver, url)
            ngdb.log_error_to_database(f"General exception: {e}")
            if driver:
                driver.quit()
            break

    return {"page_source": None} if not return_driver else {"driver": None, "page_source": None}


@app.task(name="scraper_app.initialize_selenium_helper")
def initialize_selenium_helper(task_data):
    """
    A helper function to adapt the initialize_selenium task for the task chain.

    :param task_data: The dictionary containing task arguments.
    :return: Modified task_data including the WebDriver from initialize_selenium.
    """
    try:
        print("Initializing selenium...")
        # Initialize the WebDriver
        driver = initialize_selenium()
        print("Selenium initialized...")

        # Update the 'driver' key in the task's 'prelim_args'
        if 'prelim_args' in task_data:
            task_data['prelim_args']['driver'] = driver
        else:
            task_data['prelim_args'] = {'driver': driver}

        return task_data

    except Exception as e:
        # Log the error and re-raise it
        print("Exception")
        print(f"Error initializing WebDriver: {e}")
        raise


def get_element_text_or_default_chain(restaurant_element, selectors_chains, default=None, attribute=None):
    """
    Attempts to extract text or an attribute from an element using multiple selector chains.

    This function iterates through a list of selector chains. For each chain, it tries to find an element
    matching the chain on the provided restaurant_element. If an element is found, its text or specified
    attribute is returned. If no element is found using any of the chains, a default value is returned.

    :param restaurant_element: The base element to start the search from.
    :param selectors_chains: A list of selector chains to try.
    :param default: The default value to return if no element is found.
    :param attribute: The attribute to extract from the found element, if any.

    Called by: process_grubhub_page, process_grubhub_directory_page, determine_scraper_type
    """ 
    no_element_errors = []   
    for selectors_chain in selectors_chains:
        try:
            current_element = restaurant_element
            for selector in selectors_chain:
                try:
                    current_element = current_element.find_element(By.CSS_SELECTOR, selector)
                except NoSuchElementException as e:
                    no_element_errors += [{'exception': e}, {'traceback': traceback.format_exc()}]
                    continue
            return current_element.get_attribute(attribute) if attribute else current_element.text.strip()
        except (TimeoutException, NoSuchElementException, WebDriverException) as e:
            no_element_errors += [{'exception': e}, {'traceback': traceback.format_exc()}]
            continue  # If one chain fails, try the next chain
    for no_element_error in no_element_errors:
        print("The following NoSuchElementException cases were noted:")
        print(no_element_error)
    return default  # Return default if all chains fail


def test_proxy_performance(driver, test_url="https://www.bing.com/"):
    try:
        # Set the page load timeout to 15 seconds
        driver.set_page_load_timeout(20)

        start_time = time.time()
        driver.get(test_url)
        load_time = time.time() - start_time
        print(f"Proxy Load Time: {load_time:.1f} seconds for {test_url}")
        return float(load_time)

    except TimeoutException:
        print(f"Proxy load time exceeded 15 seconds for {test_url}.")
        return float(0)  # Return 0 to indicate the proxy is too slow

    except Exception as e:
        print(f"Error testing proxy performance: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return float(0)  # Return 0 to indicate an error occurred


def get_scrapeops_headers():
    response = requests.get(f'https://headers.scrapeops.io/v1/browser-headers?api_key={SCRAPEOPS_API_KEY}')
    return response.json().get('result', [])
        

def check_proxy(driver):
    # Navigate to a test URL
    print("Checking proxy")
    driver.get("https://httpbin.org/ip")

    # Wait for the page to load and find the pre tag containing the JSON response
    wait = WebDriverWait(driver, 10)
    try:
        pre_tag = wait.until(ec.presence_of_element_located((By.TAG_NAME, 'pre')))
    except TimeoutException:
        print("A timeout exception occured while checking the proxy.")
        print(f"Full traceback: {traceback.format_exc()}")
        return

    # Extract and parse the JSON from the pre tag's text
    ip_data = json.loads(pre_tag.text)
    ip_address = ip_data['origin']

    # Check if the IP address is not one of the no_proxy addresses
    if ip_address not in ["72.89.6.16", "127.0.0.1"]:
        print(f"IP address is {ip_address}. Proxy is working.")
    else:
        print(f"IP address is {ip_address}. Proxy IS NOT working.")
    

def close_existing_drivers():
    # Attempt to close existing ChromeDriver and GeckoDriver instances
    print("Attempting to close existing drivers")
    for _ in range(2):
        for proc in psutil.process_iter(['pid', 'name']):
            print("Evaluating process: " + proc.name())
            try:
                if 'geckodriver' in proc.name() or 'firefox' in proc.name():
                    print(f"We found an existing {proc.name()} instance (PID: {proc.pid}). Attempting to close...")
                    if proc.is_running():  # Check if the process is still running
                        proc.kill()
                        try:
                            proc.wait(2)  # Increase timeout if necessary
                        except psutil.TimeoutExpired:
                            print(f"Timeout expired while waiting for {proc.name()} to terminate.")
                ps_output = subprocess.check_output(['ps', 'aux'], shell=True).decode()
                if 'firefox' in ps_output.lower():
                    try:
                        print("Firefox is running on Unix-like OS. Attempting to close...")
                        subprocess.call(['pkill', 'firefox'])
                    except Exception as e:
                        print(f"Exception {e}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                print("Exception while closing driver: ", e)
                print(f"Full traceback: {traceback.format_exc()}")
        time.sleep(2)

    # Final check to ensure all processes are killed
    for proc in psutil.process_iter():
        if 'chromedriver' in proc.name() or 'geckodriver' in proc.name() or 'firefox' in proc.name():
            print(f"Warning: An existing {proc.name()} instance still running after multiple attempts to close.")


def get_next_proxy():
    # Obtain the database connection
    db_conn = ngdb.DatabaseConnection().get_db()
    used_proxies = db_conn['used_proxies']

    # NYC residential proxies list
    proxies = [f"city.smartproxy.com:{i}" for i in range(21001, 21050)]

    # Fetch used proxies from MongoDB
    used_proxy_list = [doc['proxy'] for doc in used_proxies.find()]
    available_proxies = [proxy for proxy in proxies if proxy not in used_proxy_list]

    # If all proxies have been used, reset the list
    if not available_proxies:
        used_proxies.delete_many({})
        available_proxies = proxies

    next_proxy = available_proxies[0]
    print("Proceeding with the following proxy:", next_proxy)
    proxy_info = used_proxies.find_one({'proxy': next_proxy})
    if proxy_info:
        print(f"Last used: {proxy_info.get('last_used', 'N/A')}, Recent failures: {proxy_info.get('recent_failures', 0)}")
    else:
        print("No previous usage data available for this proxy.")
    used_proxies.insert_one({'proxy': next_proxy})

    return next_proxy


def generate_options(proxy=None):
    print("Generating Selenium options")
    options = FirefoxOptions()

    browser_headers = get_scrapeops_headers()
    selected_headers = choice(browser_headers)
    options.set_preference("general.useragent.override", selected_headers["user-agent"])  

    if proxy:
        # Proxy settings
        options.set_preference('network.proxy.type', 1)
        options.set_preference('network.proxy.http', proxy.split(":")[0])
        options.set_preference('network.proxy.http_port', int(proxy.split(":")[1]))
        options.set_preference('network.proxy.ssl', proxy.split(":")[0])
        options.set_preference('network.proxy.ssl_port', int(proxy.split(":")[1]))

    # Enable cookies during the session, but clear them on exit
    options.set_preference("privacy.clearOnShutdown.cookies", True)
    options.set_preference("privacy.clearOnShutdown.cache", True)
    options.set_preference("privacy.clearOnShutdown.offlineApps", True)
    options.set_preference("privacy.clearOnShutdown.sessions", True)

    # Cache pages to make testing faster
    options.set_preference("browser.cache.disk.enable", True)
    options.set_preference("browser.cache.memory.enable", True)
    options.set_preference("browser.cache.offline.enable", True)
    options.set_preference("network.http.use-cache", True)

    # Other settings
    # options.set_preference('permissions.default.image', 2)
    # options.set_preference('permissions.default.stylesheet', 2)
#     options.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument('--incognito')
    # options.add_argument("--headless")

    return options


def is_webdriver_active(driver):
    try:
        # Attempt a benign operation, like checking the current URL
        _ = driver.current_url
        return True
    except Exception as e:
        print(f"Exception Type: Message: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False


def initialize_selenium(retry_count=0):
    print("Closing existing drivers")
    close_existing_drivers()
        
    # Basic WebDriver options for headless mode
    proxy = get_next_proxy()
    print("Generating options")
    options = generate_options(proxy)

    try:
        print("Assigning a Firefox selenium instance")
        driver = webdriver.Firefox(options=options)
        time.sleep(random.uniform(5, 10))
        driver.delete_all_cookies()

        if random.random() < 0.1:  # random.random() returns a float between 0.0 and 1.0
            check_proxy(driver)
        
        return driver  # Return both driver and proxy
    except Exception as e:
        print(f"Exception Type: Message: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        if retry_count < WEBDRIVER_INIT_RETRIES:
            print(f"Retrying WebDriver initialization (Attempt {retry_count + 1}/{WEBDRIVER_INIT_RETRIES})")
            time.sleep(random.uniform(2, 6))
            return initialize_selenium(retry_count + 1)
        else:
            raise Exception(f"Failed to initialize WebDriver after {WEBDRIVER_INIT_RETRIES} attempts")            
            

def check_page_is_valid(page_content, driver, validity_checks):
    try:
        for check in validity_checks:
            print(f"Checking if {check['message']}")
            if not check['condition'](driver):
                print(f"Invalid page: {check['message']}")
                return False
                # Check for unusual traffic message
            if "Looks like we're missing something" in page_content or "Error code: 401" in page_content or "We couldn't cook up the page" in page_content:
                print("Grubhub blocked us.")
                return False    
                
            if "Our systems have detected unusual" in page_content:
                print("Unusual traffic detected by Google.")
                return False
            
            if "Access to this page" in page_content or "been blocked" in page_content or "PRESS & HOLD" in page_content:
                print("Grubhub blocked us.")
                return False            
            
            # Check if 'Grubhub' is present on the page
            if 'Grubhub' not in page_content:
                print("Grubhub not found on the page.")
                return False
        return True
    except Exception as e:
        print(f"Error in page validation: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False    


def create_driver_session(session_id, executor_url):
    """
    Create a new WebDriver instance and attach it to an existing session.
    """
    original_execute = RemoteWebDriver.execute
    def new_command_execute(driver, command, params=None):
        if command == "newSession":
            return {'success': 0, 'value': None, 'sessionId': session_id}
        else:
            return original_execute(driver, command, params)

    RemoteWebDriver.execute = new_command_execute
    driver = webdriver.Remote(command_executor=executor_url, desired_capabilities={})
    RemoteWebDriver.execute = original_execute

    return driver


def save_driver_session(driver):
    """
    Save the current WebDriver session.
    """
    session_id = driver.session_id
    executor_url = driver.command_executor._url
    return session_id, executor_url
