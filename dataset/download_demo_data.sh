#!/usr/bin/env bash
# ---------------------------------------------------------------------------------
# This script downloads the raw captures from the DAVIDE dataset.
# It requires a username and password for authentication.
# The script will download the files to the directory specified by the DEMO_DATA_ROOT environment variable.
# The script also has an option to clean up the zip files after extraction.
# ---------------------------------------------------------------------------------
# Usage: bash ./dataset/download_demo_data.sh <username> <password> [--clean]
# ---------------------------------------------------------------------------------
""""""
# Set help option
if [[ $1 == "-h" || $1 == "--help" ]]; then
    sed '/^[^#]/q' "$0"
    exit 0
fi

# Check if DEMO_DATA_ROOT is set
if [ -z "$DEMO_DATA_ROOT" ]; then
  echo "Error: DEMO_DATA_ROOT environment variable is not set."
  exit 1
fi

# Set credentials
USERNAME="$1"
PASSWORD="$2"
shift 2  # Shift arguments so we can check for --clean
# Check if username and password are provided
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
  echo "Error: Username and password are required."
  echo "Usage: bash ./dataset/download_demo_data.sh <username> <password> [--clean]"
  exit 1
fi

# Check if the --clean flag is provided
CLEAN=false
if [ "$1" = "--clean" ]; then
  CLEAN=true
fi

# Create and switch to the workspace directory
mkdir -p "$DEMO_DATA_ROOT"
cd "$DEMO_DATA_ROOT"

# Demo data link
demo_data_link="https://davide.rd.tuni.fi/davide-dataset/demo/demo-data.zip"

# Download the demo data
curl -u "${USERNAME}:${PASSWORD}" -O "${demo_data_link}"

# Unzip the downloaded file
unzip demo-data.zip 

# Optionally remove the zip files
if [ "$CLEAN" = true ]; then
  rm demo-data.zip
fi

# Download license file
curl -u "${USERNAME}:${PASSWORD}" -O "https://davide.rd.tuni.fi/davide-dataset/LICENSE.txt"