# mongodb.Dockerfile
FROM mongo:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Get requirements to install in container
COPY requirements.txt /usr/src/app/

# Install additional packages, including ss
RUN apt-get update && \
    apt-get install -y net-tools iproute2

# Copy your application code to the container
COPY . /usr/src/app

# Ensure proper permissions for entrypoint.sh
RUN chmod +x entrypoint.sh init-db.sh

# Grant permissions to the mongodb user for /data/db
RUN chown -R mongodb:mongodb /data/db

# Command to ensure stays running
CMD ["tail", "-f", "/dev/null"]
