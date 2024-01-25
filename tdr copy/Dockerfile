# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /usr/src/app

# Get requirements to install in container
COPY requirements.txt /usr/src/app/

# Install build-essential for compiling dependencies
RUN apt-get update && \
    apt-get install -y build-essential net-tools

# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code to the container
COPY . /usr/src/app

# Ensure proper permissions for entrypoint.sh
RUN chmod +x entrypoint.sh init-db.sh start.sh

## Disable root access after no longer needed
#RUN adduser --disabled-password --gecos '' myuser
#USER myuser

# Python environment variable
ENV PYTHONPATH=/usr/src/app

# Initialize celery workers
# Run scrape_runner.py on container startup
CMD ["./start.sh"]
