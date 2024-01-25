#!/bin/bash
# entrypoint.sh
echo "Starting MongoDB setup..."

# Create a user (mongodb is the default MongoDB user)
# Only include this if your specific setup requires creating a new user
adduser --disabled-password --gecos '' mongodb

# Grant permissions to the mongodb user for /data/db
# Only include this if your setup requires setting permissions manually
chown -R mongodb:mongodb /data/db

# Run any necessary initialization scripts
# Include your script here if you have any initialization logic
/init-db.sh

# Start MongoDB in the foreground
# This command should be the last one and will keep the container running
exec mongod --bind_ip_all

echo "entrypoint.sh has completed running."
