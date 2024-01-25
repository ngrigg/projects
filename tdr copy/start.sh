#!/bin/bash
# start.sh
echo "starting celery"
celery -A celery_app.app worker --loglevel=info &
echo "starting python scrape runner"
python /usr/src/app/scrape_runner.py &
echo "waiting for background processes"
wait
echo "running tail to ensure persistence"
tail -f /dev/null
echo "complete"
