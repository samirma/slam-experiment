# Dockerfile for the main Flask web application
FROM python:3.10-slim

# Install Docker client CLI for the main app to be able to launch Docker sibling containers
# This is necessary because script_runner.py uses the Docker SDK.
# We need Docker CLI and access to Docker socket.
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*
# Alternatively, could use docker-in-docker, but sharing host's docker socket is more common for this pattern.

WORKDIR /app

# Copy requirements for the main web application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files needed for the script runner image build process
# Dockerfile.runner will be at /app/Dockerfile.runner
COPY Dockerfile.runner .
# app/runner_requirements.txt will be at /app/app/runner_requirements.txt
# This is relative to WORKDIR /app
COPY app/runner_requirements.txt ./app/runner_requirements.txt

# Copy the rest of the application code
# app module (Flask app) will be at /app/app/
COPY app ./app
# crewai_reference.md will be at /app/crewai_reference.md
COPY crewai_reference.md .

EXPOSE 5000

# Command to run the application
# `python -m app.main` will execute /app/app/main.py as the main script of the `app` package.
# WORKDIR is /app, so Python will find the 'app' package in the 'app' subdirectory.
CMD ["python", "-m", "app.main"]
