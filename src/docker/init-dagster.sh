#!/bin/bash
set -e

echo "Initializing Dagster..."

# Wait for the database directory to be available
DAGSTER_HOME=${DAGSTER_HOME:-/data/dagster_home}
mkdir -p "$DAGSTER_HOME"

# Initialize the database if it doesn't exist
if [ ! -f "$DAGSTER_HOME/dagster.db" ]; then
    echo "Creating new Dagster database..."
    dagster instance bootstrap
fi

# Handle migrations
echo "Running database migrations..."
dagster instance migrate

# Start Dagster
echo "Starting Dagster services..."
dagster-daemon run &
dagit -h 0.0.0.0 -p 3000 