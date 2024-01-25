#!/usr/bin/env python
# coding: utf-8

# !pip install fake-useragent
# !pip install selenium-wire
# !pip install webdriver-manager
# !pip install pymongo
# !pip install --upgrade webdriver-manager
# !pip install html5lib
from importlib.metadata import distributions
import os
import subprocess
from datetime import datetime
import csv
import traceback
import threading
import redis
from time import sleep


# Global variables
LARGE_CHAINS_FILE = 'large_chains.txt'
NEIGHBORHOODS_FILE = 'grub_neighborhoods.csv'
CUISINES_FILE = 'grub_cuisines.txt'
SCRAPEOPS_API_KEY = 'b77ca529-e955-499d-8183-b258251ee3c4'


# def check_and_manage_docker_redis():
#     try:
#         subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Status}}', 'tdr-redis-1'])
#         print("Redis Docker container tdr-redis-1 is running.")
#         subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Status}}', 'tdr-worker-1'])
#         print("Celery Worker Docker container tdr-worker-1 is running.")
#     except subprocess.CalledProcessError:
#         print("Redis Docker container is not running. Attempting to start...")
#         try:
#             subprocess.call(['docker', 'run', '-d', '--name', 'tdr-redis-1', '-p', '6379:6379', 'redis:latest'])
#             print("New Redis Docker container started.")
#             subprocess.call(['docker', 'run', '-d', '--name', 'tdr-worker-1', 'tdr-worker'])
#             print("New Redis Worker container started.")
#         except Exception as e:
#             print(f"Failed to start Redis Docker container: {e}")
#             raise


def read_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        output = [row for row in reader]
    return output


def read_txt(file_path):
    with open(file_path, 'r') as file:
        txt = [line.strip() for line in file if line.strip()]
    return txt


def quote_if_multiple_words(text):
    return f'"{text}"' if ' ' in text else text


def format_relative_time(timestamp):
    diff = datetime.now() - timestamp
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds // 3600 > 0:
        return f"{diff.seconds // 3600} hours ago"
    else:
        return f"{diff.seconds // 60} minutes ago"


def load_cuisines_from_file(file_path):
    """
    Loads a list of cuisines from a specified file.

    This function opens the given file and reads each line, assuming each line represents a cuisine.
    It returns a set of cuisines read from the file.

    :param file_path: The file path to read cuisines from.

    :return: A set of cuisines.

    Called by: scrape_grubhub_google_cache_urls
    """
    try:
        with open(file_path, 'r') as file:
            cuisines = {line.strip().split()[0].lower() for line in file.readlines()}
        return cuisines
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        print("Exception", e)
        print(f"Full traceback: {traceback.format_exc()}")
        return set()


def take_screenshot(driver, reason="Error", auto_display=True, timeout=15):
    screenshot_taken = [False]

    def screenshot_logic():
        try:
            print(f"Attempting screenshot due to: {reason}")
            images_dir = 'images'
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{images_dir}/{reason}_screenshot_{timestamp}.png"
            if driver.save_screenshot(filename):
                print(f"Screenshot saved: {filename}")
                screenshot_taken[0] = True
                if auto_display:
                    # Open the image file with the default image viewer
                    if os.name == 'nt':  # Windows
                        os.startfile(filename)
                    elif os.name == 'posix':  # macOS and Linux
                        subprocess.call(['open', filename])
            else:
                print("Failed to save screenshot.")
        except Exception as e:
            print(f"Error taking screenshot: {e}")

    screenshot_thread = threading.Thread(target=screenshot_logic)
    screenshot_thread.start()
    screenshot_thread.join(timeout)

    if screenshot_thread.is_alive():
        print("Screenshot attempt timed out.")

    return screenshot_taken[0]


def generate_requirements_from_imports(import_list):
    """
    Generates a requirements.txt content from a list of imports with their versions.

    :param import_list: List of imported package names.
    :return: String with requirements.txt content.
    """

    # Get installed package versions
    installed_packages = {dist.metadata['Name']: dist.version for dist in distributions()}

    # Generate requirements content
    requirements_content = ""
    for package in import_list:
        if package in installed_packages:
            requirements_content += f"{package}=={installed_packages[package]}\n"
        else:
            requirements_content += f"# {package} not found in installed packages\n"

    return requirements_content


# if you're testing on your local machine, then host is localhost
# if you are testing on docker, then host is redis
def check_redis_connection():
    try:
        client = redis.StrictRedis(host='redis', port=6379, db=0)
        pong = client.ping()
        print("Redis connection successful:", pong)
        print("Running on localhost port 6379 db 0")
    except redis.ConnectionError:
        print("Failed to connect to Redis... retrying...")
        sleep(5)
        check_redis_connection()
