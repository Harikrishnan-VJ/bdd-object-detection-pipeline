#!/bin/bash

# Define variables
host_path="/home/user/hari/test/bdd-object-detection-pipeline/data_analysis"
container_path="/app"
image_name="bdd-analysis"
image_tag="v1.0"
container_name="bdd-analysis-container"

# Path to the systemd service unit file
# For installing as a service (if needed)
# service_file="docker_monitor.service"

echo "Building Container..."
# Building image
docker build --no-cache -t "$image_name:$image_tag" -f dockerfiles/Dockerfile .

echo "Building Container... Done"

# For installing as aservice (if needed)
# echo "Installing systemd service..."
# # Copy the systemd service unit file to the appropriate location
# echo "pass1@3" | sudo -S  cp "$service_file" /etc/systemd/system/
    
# # Reload systemd to read the new service unit file
# echo "pass1@3" | sudo -S systemctl daemon-reload


# # Enable and start the service
# echo "pass1@3" | sudo -S systemctl enable docker_monitor.service
# echo "pass1@3" | sudo -S systemctl start docker_monitor.service

# echo "Installing systemd service... Done"

echo "Running Container..."
# Running container
docker run -d -v $host_path:$container_path --name $container_name -it  $image_name":"$image_tag