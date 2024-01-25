# /task_mgmt/celery_app.py

from celery import Celery
from celery.schedules import crontab


class CeleryConfig:
    # Use Redis as the broker, specify the Redis connection URL
    # Replace 'localhost' with the hostname or IP address of your Redis container
    # Include your Redis password in the URL
    # PASSWORD = 'mypassword'
    password = ''
    # broker_url = f'redis://:{password}@redis:6379/0'
    broker_url = f'redis://redis:6379/0'
    # broker_url = f'redis://localhost:6379/0'
    # result_backend = f'redis://:{password}@redis:6379/0'
    result_backend = f'redis://redis:6379/0'
    # result_backend = f'redis://localhost:6379/0'

    # Set the result backend to use Redis as well
    # Replace 'localhost' with the hostname or IP address of your Redis container
    # Include your Redis password in the URL

    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    timezone = 'UTC'
    enable_utc = True
    task_track_started = True
    task_soft_time_limit = 300  # in seconds

    worker_hijack_root_logger = False
    worker_log_color = False
    worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
    worker_task_log_format = "[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s"


# Initialize Celery application
app = Celery('scraper_app')

# Load configurations from config.py
app.config_from_object(CeleryConfig)

# If you have tasks in different files, you can use autodiscover
app.autodiscover_tasks(force=True)
app.autodiscover_tasks(['scrapers', 'task_mgmt'])

app.conf.beat_schedule = {
    'dispatch-tasks-every-minute': {
        'task': 'task_mgmt.tasks.dispatch_tasks',
        'schedule': crontab(minute='*'),  # Adjust as needed
    },
    'check-docker-redis-every-5-minutes': {
        'task': 'task_mgmt.tasks.check_docker_redis_task',
        'schedule': crontab(minute='*/5'),  # Adjust as needed
    },
}
