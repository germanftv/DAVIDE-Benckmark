#!/usr/bin/env bash
# ---------------------------------------------------------------------------------
# This script downloads the official released version of the DAVIDE dataset.
# It requires a username and password for authentication.
# The script will download the files to the directory specified by the DATASET_ROOT environment variable.
# Optionally:
# - If the --split flag is provided, it will download the specific split of the dataset: train, test, or train+test. Default is train+test.
# - If the --mono_depth flag is provided, it will download the precomputed monocular depth maps.
# - If the --meta_data flag is provided, it will download the metadata files.
# - If the --clean flag is provided, it will remove the downloaded zip files after extraction.
# ---------------------------------------------------------------------------------
# Usage: bash ./dataset/download_davide_dataset.sh <username> <password> [--split <split>] [--mono_depth] [--meta_data] [--clean]
# ---------------------------------------------------------------------------------
""""""
# Set help option
if [[ $1 == "-h" || $1 == "--help" ]]; then
    sed '/^[^#]/q' "$0"
    exit 0
fi

# Check if DATASET_ROOT is set
if [ -z "$DATASET_ROOT" ]; then
  echo "Error: DATASET_ROOT environment variable is not set."
  exit 1
fi

# Set credentials
USERNAME="$1"
PASSWORD="$2"
shift 2  # Shift arguments so we can check for flags
# Check if username and password are provided
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
  echo "Error: Username and password are required."
  echo "Usage: bash ./dataset/download_davide_dataset.sh <username> <password> [--split <split>] [--mono_depth] [--meta_data] [--clean]"
  exit 1
fi

# Set split
SPLIT="train+test"  # Default split
if [ "$1" = "--split" ]; then
  SPLIT="$2"
  shift 2  # Shift again to remove the split argument
fi
# Check if the split argument is valid
if [ "$SPLIT" != "train" ] && [ "$SPLIT" != "test" ] && [ "$SPLIT" != "train+test" ]; then
  echo "Error: Invalid split argument. Use 'train', 'test', or 'train+test'."
  exit 1
fi

# Set mono depth flag
MONO_DEPT=false
if [ "$1" = "--mono_depth" ]; then
  MONO_DEPT=true
  shift  # Shift to remove the mono depth argument
fi

# Set meta data flag
META_DATA=false
if [ "$1" = "--meta_data" ]; then
  META_DATA=true
  shift  # Shift to remove the meta data argument
fi

# Set clean flag
CLEAN=false
if [ "$1" = "--clean" ]; then
  CLEAN=true
fi

# Create and switch to the workspace directory
mkdir -p "$DATASET_ROOT"
cd "$DATASET_ROOT"

# Download the DAVIDE dataset
if [ "$SPLIT" = "train" ] || [ "$SPLIT" = "train+test" ]; then
    davide_train_root="https://davide.rd.tuni.fi/davide-dataset/train"

    # Make a list of the files to download: DAVIDE-train.z01, DAVIDE-train.z02, ..., DAVIDE-train.z40 , DAVIDE-train.zip
    davide_train_files=()
    for i in $(seq 1 40); do
        davide_train_files+=("${davide_train_root}/DAVIDE-train.z$(printf '%02d' "$i")")
    done
    davide_train_files+=("${davide_train_root}/DAVIDE-train.zip")
    
    # Download the files
    for file in "${davide_train_files[@]}"; do
        curl -u "${USERNAME}:${PASSWORD}" -O "${file}"
    done

    # Unzip the files
    zip -s 0 DAVIDE-train.zip --out DAVIDE-train_combined.zip
    unzip DAVIDE-train_combined.zip

    # Rename DAVIDE-train to train
    mv DAVIDE-train train

    # Download the monocular depth maps if requested
    if [ "$MONO_DEPT" = true ]; then
        cd "$DATASET_ROOT/train" 
        davide_train_mono_depth_root="https://davide.rd.tuni.fi/davide-dataset/train"
        davide_train_mono_depth_files=()
        for i in $(seq 1 2); do
            davide_train_mono_depth_files+=("${davide_train_mono_depth_root}/mono-depth.z$(printf '%02d' "$i")")
        done
        davide_train_mono_depth_files+=("${davide_train_mono_depth_root}/mono-depth.zip")
        
        # Download the files
        for file in "${davide_train_mono_depth_files[@]}"; do
            curl -u "${USERNAME}:${PASSWORD}" -O "${file}"
        done

        # Unzip the files
        zip -s 0 mono-depth.zip --out DAVIDE-train-mono-depth_combined.zip
        unzip DAVIDE-train-mono-depth_combined.zip

        # Optionally remove the zip files
        if [ "$CLEAN" = true ]; then
            rm -rf *mono-depth*.z*
        fi

        # Move content to the main directory
        mv mono-depth/* ./
        rm -rf mono-depth

        # Go back to the main directory
        cd "$DATASET_ROOT"
    fi
fi

if [ "$SPLIT" = "test" ] || [ "$SPLIT" = "train+test" ]; then
    davide_test_root="https://davide.rd.tuni.fi/davide-dataset/test"

    # Make a list of the files to download: DAVIDE-test.z01, DAVIDE-test.z02, ..., DAVIDE-test.z07 , DAVIDE-test.zip
    davide_test_files=()
    for i in $(seq 1 7); do
        davide_test_files+=("${davide_test_root}/DAVIDE-test.z$(printf '%02d' "$i")")
    done
    davide_test_files+=("${davide_test_root}/DAVIDE-test.zip")
    
    # Download the files
    for file in "${davide_test_files[@]}"; do
        curl -u "${USERNAME}:${PASSWORD}" -O "${file}"
    done

    # Unzip the files
    zip -s 0 DAVIDE-test.zip --out DAVIDE-test_combined.zip
    unzip DAVIDE-test_combined.zip

    # Rename DAVIDE-test to test
    mv DAVIDE-test test

    # Download the monocular depth maps if requested
    if [ "$MONO_DEPT" = true ]; then
        cd "$DATASET_ROOT/test"
        davide_test_mono_depth_root="https://davide.rd.tuni.fi/davide-dataset/test"
        davide_test_mono_depth_file="${davide_test_mono_depth_root}/mono-depth.zip"
        
        # Download the file
        curl -u "${USERNAME}:${PASSWORD}" -O "${davide_test_mono_depth_file}"

        # Unzip the files
        unzip mono-depth.zip
        # Optionally remove the zip files
        if [ "$CLEAN" = true ]; then
            rm -rf *mono-depth*.z*
        fi

        # Move content to the main directory
        mv mono-depth/* ./
        rm -rf mono-depth

        # Go back to the main directory
        cd "$DATASET_ROOT"
    fi
fi

# Download the metadata files if requested
if [ "$META_DATA" = true ]; then
    wget --recursive --no-parent --reject "index.html*" --cut-dirs=2 --no-host-directories --user="${USERNAME}" --password="${PASSWORD}" https://davide.rd.tuni.fi/davide-dataset/meta-data/
fi

# Optionally remove the zip files
if [ "$CLEAN" = true ]; then
    rm -rf *DAVIDE-*.z*
fi

# Download license file
curl -u "${USERNAME}:${PASSWORD}" -O "https://davide.rd.tuni.fi/davide-dataset/LICENSE.txt"