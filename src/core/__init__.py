"""Core package for log parsing and analysis using Dagster.

This module initializes the Dagster definitions including assets and jobs
for the on-demand log parsing pipeline. The pipeline is triggered through
the Streamlit UI when users upload log files for analysis.
"""

from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job
)

from . import assets

# Load all assets from the assets module
all_assets = load_assets_from_modules([assets])

# Define the log processing job that will materialize all assets
log_processing_job = define_asset_job(
    name="log_processing_job",
    selection=all_assets
)

# Create the Dagster definitions
defs = Definitions(
    assets=all_assets,
    jobs=[log_processing_job]
) 