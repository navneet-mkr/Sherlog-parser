#!/bin/bash
set -e

echo "Initializing Dagster..."

# Ensure DAGSTER_HOME is set
DAGSTER_HOME=${DAGSTER_HOME:-/data/dagster_home}
export DAGSTER_HOME

# Create necessary directories
mkdir -p "$DAGSTER_HOME"
mkdir -p "$DAGSTER_HOME/storage"
mkdir -p "$DAGSTER_HOME/compute_logs"

# Remove any existing config
rm -f "$DAGSTER_HOME/dagster.yaml"

# Create fresh Dagster instance config
echo "Creating Dagster instance configuration..."
cat > "$DAGSTER_HOME/dagster.yaml" << EOL
telemetry:
  enabled: false

local_artifact_storage:
  module: dagster.core.storage.root
  class: LocalArtifactStorage
  config:
    base_dir: "$DAGSTER_HOME/storage"

compute_logs:
  module: dagster.core.storage.local_compute_log_manager
  class: LocalComputeLogManager
  config:
    base_dir: "$DAGSTER_HOME/compute_logs"

run_storage:
  module: dagster.core.storage.runs
  class: SqliteRunStorage
  config:
    base_dir: "$DAGSTER_HOME"

event_log_storage:
  module: dagster.core.storage.event_log
  class: SqliteEventLogStorage
  config:
    base_dir: "$DAGSTER_HOME"

schedule_storage:
  module: dagster.core.storage.schedules
  class: SqliteScheduleStorage
  config:
    base_dir: "$DAGSTER_HOME"

scheduler:
  module: dagster.core.scheduler
  class: DagsterDaemonScheduler

run_coordinator:
  module: dagster.core.run_coordinator
  class: QueuedRunCoordinator

run_launcher:
  module: dagster.core.launcher
  class: DefaultRunLauncher
EOL

echo "Created config file at: $DAGSTER_HOME/dagster.yaml"
cat "$DAGSTER_HOME/dagster.yaml"

# Handle migrations
echo "Running database migrations..."
dagster instance migrate

# Start Dagster
echo "Starting Dagster services..."
dagster-daemon run -m src.core &
dagster-webserver -h 0.0.0.0 -p 3000 -m src.core 