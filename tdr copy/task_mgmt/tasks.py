# /task_mgmt/tasks.py

from celery import chain
from celery_app import app
from celery.utils.log import get_task_logger
from utils import db_manager as ngdb
import scrapers
from scrapers import gh_category_scraper as ghcat, gh_restaurant_scraper as ghrest, goog_search_scraper as g_scraper
from scrapers import scraper_utils as s_utils, goog_search_scraper, gh_category_scraper, gh_restaurant_scraper
import traceback
import logging

# Logger for Celery tasks
logger = get_task_logger(__name__)

MAX_ERROR_THRESHOLD = 1  # Max retries before marking a task as failed


def find_high_priority_tasks(find_params={'scrape_attempted': False}, task_load_limit=50):
    """
    Finds and prioritizes high-priority tasks from the database based on the given criteria.

    :param find_params: Criteria for selecting tasks. Default is {'scrape_attempted': False}.
    :param task_load_limit: Maximum number of tasks to load.
    :return: A list of prioritized task records.
    """
    # Fetch tasks based on criteria using the find_all function
    print("Finding high priority tasks...")
    tasks = ngdb.find_all('tasks', find_params)
    print(f"Found {len(tasks)} tasks based on criteria: {find_params}")

    # Prioritize tasks based on your business logic
    # For now no prioritization is occurring
    prioritized_tasks = tasks
    print(f"Prioritizing tasks. Limit set to {task_load_limit}")

    # Return the top 'task_load_limit' tasks
    return prioritized_tasks[:task_load_limit]


def prepare_task_arguments(task):
    """
    Prepares the necessary arguments for task execution, including args, prelim_args, etc.

    :param task: A dictionary representing the task document.
    :return: A dictionary formatted for Celery task dispatch.
    """
    print(f"Preparing task arguments for task: {task.get('_id')}")
    # Extract necessary fields from the task
    last_step = task.get('last_step')
    next_step = task.get('next_step')
    params = task.get('params', {})
    task['_id'] = str(task['_id'])

    # Determine task type and prepare arguments
    if last_step and next_step:
        args, prelim_args, prelim_steps, kwargs = get_args_and_prelim_steps_for_task(task, params, last_step, next_step)
    else:
        print("ERROR: Task type not recognized or necessary fields missing")
        return None

    # Format the task data for Celery
    celery_task_data = {
        'id': task['_id'],
        'function': next_step,
        'args': args,
        'prelim_args': prelim_args,
        'prelim_steps': prelim_steps,
        'kwargs': kwargs
    }
    print(f"The following arguments were prepared for Celery {celery_task_data}")
    return celery_task_data


def get_args_and_prelim_steps_for_task(task, params, last_step, next_step):
    if last_step == "get_starting_google_search_urls" and next_step == "get_google_cached_urls_from_google_search_urls":
        print("evaluating args for next step get_google_cached_urls_from_google_search_urls")
        args = {
            'task': task,
            'driver': None,  # process first goes to fetch_and_process_page, which sets this parameter
            'search_url': params['search_url'],
            'path_ext_pattern': 'grubhub\.com/delivery/([^\s\\?]+)(?:\?pageNum=(\d+))?(?:\\\\u)?',
            'must_contain': ['grubhub'],
            'must_not_contain': ['yelp'],
            'prefix': 'https://www.grubhub.com/delivery/',
            'retry_count': 0,
            'last_step': last_step,
            'library': 'g_scraper'
        }
        prelim_args = {'driver': None,
                       'url': params['search_url'],
                       'return_driver': True}
        prelim_steps = {
            'fetch_and_process_page': True
        }
        kwargs = {}
        return args, prelim_args, prelim_steps, kwargs
    elif last_step == "get_google_cached_urls_from_google_search_urls" and next_step == "get_gh_restaurant_links_from_google_cache_gh_category_pages":
        print("evaluating args for next step get_gh_restaurant_links_from_google_cache_gh_category_pages")
        args = {
            'task': task,
            'driver': None,
            'search_url': params['search_url'],
            'retry_count': 0,
            'last_step': last_step,
            'library': 'ghcat'
        }
        prelim_args = {'driver': None,
                       'url': params['search_url'],
                       'return_driver': True}
        prelim_steps = {
            'fetch_and_process_page': True
        }
        kwargs = {}
        return args, prelim_args, prelim_steps, kwargs
    elif last_step == "get_gh_restaurant_links_from_google_cache_gh_category_pages" and next_step == "get_google_cached_urls_from_google_search_urls":
        print("evaluating args for last step get_gh_restaurant_links_from_google_cache_gh_category_pages")
        args = {
            'task': task,
            'driver': None,
            'search_url': params['search_url'],
            'retry_count': 0,
            'last_step': last_step,
            'library': 'g_scraper',
        }
        prelim_args = {'driver': None,
                       'url': params['search_url'],
                       'return_driver': True}
        prelim_steps = {
            'fetch_and_process_page': True
        }
        kwargs = {}
        return args, prelim_args, prelim_steps, kwargs
    elif last_step == "get_google_cached_urls_from_google_search_urls" and next_step == "get_gh_restaurant_data_from_google_cache_gh_restaurant_page":
        print("evaluating args for next step of get_gh_restaurant_data_from_google_cache_gh_restaurant_page")
        args = {
            'task': task,
            'driver': None,  # Will be set by fetch_and_process_page
            'search_url': params['search_url'],
            'retry_count': 0,
            'last_step': last_step,
            'library': 'ghrest',
        }
        prelim_args = {
            'driver': None,
            'url': params['search_url'],  # URL obtained from params
            'return_driver': True
        }
        prelim_steps = {
            'fetch_and_process_page': True
        }
        kwargs = {}
        return args, prelim_args, prelim_steps, kwargs
    else:
        print("ERROR DID NOT FIND ARGUMENTS APPROPRIATE FOR LOADED TASK")
        return None


@app.task(name="scraper_app.dispatch_tasks")
def dispatch_tasks(find_params, task_load_limit=50):
    # try:
    #     clear_queue()
    # except IndexError as ie:
    #     print("Nothing in queue to pop, catching IndexError", ie)
    # print("Queue cleared before dispatching new tasks.")
    tasks_to_dispatch = find_high_priority_tasks(find_params, task_load_limit)
    # display_queue_status()
    print(f"Dispatching {len(tasks_to_dispatch)} tasks...")
    logging.info("Task is running")
    for task in tasks_to_dispatch:
        print(f"Dispatching the following task: {task}")
        # Prepare the arguments for the task chain
        prepared_task_data = prepare_task_arguments(task)

        # Initiate the task chain with the prepared data
        print(f"Initiating task chain for task {prepared_task_data}")
        try:
            initiate_task_chain.apply_async(args=[prepared_task_data])
        except Exception as e:
            print("Exception", e)
            print(f"Full traceback: {traceback.format_exc()}")
            logging.error("Exception", e)
            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise

    print(f"Dispatched {len(tasks_to_dispatch)} tasks to Celery.")


@app.task(name="scraper_app.initiate_task_chain")
def initiate_task_chain(task):
    logger.info("Initiate task chain")
    return chain(
        s_utils.initialize_selenium_helper.s(task),
        s_utils.fetch_and_process_page_helper.s(task),
        execute_final_task.s(task)  # execute_task is now the final step in the chain
    ).apply_async()


@app.task(bind=True, name="scraper_app.execute_final_task")
def execute_final_task(task_data):
    """
    Executes a given task with the provided task data.
    """
    try:
        # Log the start of task execution
        logger.info(f"Starting task execution for {task_data['id']}")

        # Extract and handle complex argument structures from task_data
        args = task_data['args']

        # Retrieve the final task function
        final_task_function = getattr(args['library'], task_data['function'])
        print(f"Executing function: {task_data['function']} for task ID: {task_data['id']}")

        # Execute the function
        result = final_task_function(args)

        # Handle task completion
        handle_task_completion(task_data['id'], result)

        # Log the end of task execution
        logger.info(f"Task {task_data['id']} completed successfully")

        return result

    except Exception as e:
        # Error handling and retry logic
        task_data['error_count'] = task_data.get('error_count', 0) + 1
        logger.error(f"Error executing task {task_data['id']}: {e}")
        logger.error(traceback.format_exc())

        if task_data['error_count'] <= MAX_ERROR_THRESHOLD:
            # Retry the task once
            logger.info(f"Retrying task {task_data['id']}")
            raise self.retry(exc=e)
        else:
            # Maximum retries exceeded, mark as failed
            handle_task_error(task_data['id'], e)
            logger.error(f"Task {task_data['id']} failed after retries")
            return {'status': 'failed', 'error': str(e)}


def handle_task_completion(task_id, result):
    """
    Handles actions to be performed upon task completion.
    This function is called when a task is successfully completed.

    :param task_id: The unique identifier of the task.
    :param result: The result of the task execution.
    """
    print(f"Handling task completion for task ID: {task_id}")
    # Update the task's status in the database to 'completed'
    collection_name = 'tasks'
    ngdb.update_database_task_status(collection_name, task_id, "completed")

    # Log the successful completion to the console
    print(f"Task {task_id} successfully completed with result: {result}")

    # If required, additional actions can be performed here, like notifying other systems or triggering dependent tasks


def handle_task_error(task_id, error):
    """
    Handles actions to be performed upon task error.
    This function is called when a task encounters an error and exceeds the retry limit.

    :param task_id: The unique identifier of the task.
    :param error: The error encountered during task execution.
    """
    print(f"Handling task error for task ID: {task_id}")
    # Update the task's status in the database to 'failed' after the retry limit is reached
    collection_name = 'tasks'
    ngdb.update_database_task_status(collection_name, task_id, "failed")

    # Log the error to the console
    print(f"Task {task_id} failed with error: {error}")

    # If required, additional actions can be taken here, such as alerting or logging for further analysis


# def clear_queue():
#     """
#     Clears all scheduled tasks from the queue.
#
#     :return: None
#     """
#     i = app.control.inspect()
#     scheduled_tasks = i.scheduled()
#     for worker, tasks in scheduled_tasks.items():
#         for task in tasks:
#             AsyncResult(task['request']['id']).revoke(terminate=True)
#
#
# def display_queue_status():
#     """
#     Displays the current status of the task queue.
#     """
#     print("Displaying the current status of the task queue.")
#     i = app.control.inspect()
#
#     active_tasks = i.active()
#     scheduled_tasks = i.scheduled()
#
#     print("Active Tasks:")
#     for worker, tasks in active_tasks.items():
#         print(f"  Worker: {worker}")
#         for task in tasks:
#             print(f"    Task: {task['name']} (ID: {task['id']})")
#
#     print("\nScheduled Tasks:")
#     for worker, tasks in scheduled_tasks.items():
#         print(f"  Worker: {worker}")
#         for task in tasks:
#             print(f"    Task: {task['request']['name']} (ID: {task['request']['id']})")
