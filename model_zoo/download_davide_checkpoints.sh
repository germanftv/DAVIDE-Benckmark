#!/usr/bin/env bash
# ---------------------------------------------------------------------------------
# This script downloads pre-trained model checkpoints for the DAVIDE dataset.
# It requires a username and password for authentication.
# The script will download the files to the directory specified by the DATASET_ROOT environment variable.
# Optionally:
# - If the --exp flag is provided, it will download the specific experiment checkpoints. Options are: "E0", "E1", "E2", "all". Default is "all".
# - If the --ckpt_dir flag is provided, it will download the checkpoints to the specified directory. Default is "./model_zoo/davide_checkpoints".
# ---------------------------------------------------------------------------------
# Usage: bash ./model_zoo/download_davide_checkpoints.sh <username> <password> [--exp <exp>] [--ckpt_dir <ckpt_dir>]
# ---------------------------------------------------------------------------------
""""""
# Set help option
if [[ $1 == "-h" || $1 == "--help" ]]; then
    sed '/^[^#]/q' "$0"
    exit 0
fi

# Set credentials
USERNAME="$1"
PASSWORD="$2"
shift 2  # Shift arguments so we can check for flags
# Check if username and password are provided
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
  echo "Error: Username and password are required."
  echo "Usage: bash ./model_zoo/download_davide_checkpoints.sh <username> <password> [--exp <exp>] [--ckpt_dir <ckpt_dir>]"
  exit 1
fi

# Set experiments to download
EXP="all"  # Default experiment
if [ "$1" = "--exp" ]; then
  EXP="$2"
  shift 2  # Shift again to remove the exp argument
fi
# Check if the exp argument is valid
if [ "$EXP" != "E0" ] && [ "$EXP" != "E1" ] && [ "$EXP" != "E2" ] && [ "$EXP" != "all" ]; then
  echo "Error: Invalid exp argument. Use 'E0', 'E1', 'E2', or 'all'."
  exit 1
fi

# Set checkpoint directory
CKPT_DIR="./model_zoo/davide_checkpoints"  # Default checkpoint directory
if [ "$1" = "--ckpt_dir" ]; then
  CKPT_DIR="$2"
  shift 2  # Shift again to remove the ckpt_dir argument
fi

# Create and switch to the checkpoint directory
mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"

# Download the DAVIDE checkpoints
if [ "$EXP" = "E0" ] || [ "$EXP" = "all" ]; then
    # Create a directory for E0 checkpoints and switch to it
    mkdir -p "E0_depth_quality"
    cd "E0_depth_quality"
    # Set the root URL for the DAVIDE checkpoints
    davide_ckpt_root="https://davide.rd.tuni.fi/checkpoints/E0_depth_quality"
    # List of files to download:
    davide_ckpt_files=(
        "Shiftnet_mono_depth_blur.pth"
        "Shiftnet_mono_depth_sharp.pth"
        "Shiftnet_sensor_depth.pth"
    )
    # Download the files
    for file in "${davide_ckpt_files[@]}"; do
        curl -u "${USERNAME}:${PASSWORD}" -O "${davide_ckpt_root}/${file}"
    done

    # Go back to the checkpoint directory
    cd ..
fi

if [ "$EXP" = "E1" ] || [ "$EXP" = "all" ]; then
    # Create a directory for E1 checkpoints and switch to it
    mkdir -p "E1_depth_impact"
    cd "E1_depth_impact"
    # Set the root URL for the DAVIDE checkpoints
    davide_ckpt_root="https://davide.rd.tuni.fi/checkpoints/E1_depth_impact"
    # List of files to download:
    davide_ckpt_files=(
        "RGBD_Shiftnet_T01_seed10.pth"
        "RGBD_Shiftnet_T01_seed13.pth"
        "RGBD_Shiftnet_T01_seed17.pth"
        "RGBD_Shiftnet_T03_seed10.pth"
        "RGBD_Shiftnet_T03_seed13.pth"
        "RGBD_Shiftnet_T03_seed17.pth"
        "RGBD_Shiftnet_T05_seed10.pth"
        "RGBD_Shiftnet_T05_seed13.pth"
        "RGBD_Shiftnet_T05_seed17.pth"
        "RGBD_Shiftnet_T07_seed10.pth"
        "RGBD_Shiftnet_T07_seed13.pth"
        "RGBD_Shiftnet_T07_seed17.pth"
        "RGBD_Shiftnet_T09_seed10.pth"
        "RGBD_Shiftnet_T09_seed13.pth"
        "RGBD_Shiftnet_T09_seed17.pth"
        "RGBD_Shiftnet_T11_seed10.pth"
        "RGBD_Shiftnet_T11_seed13.pth"
        "RGBD_Shiftnet_T11_seed17.pth"
        "baseRGB_Shiftnet_T01_seed10.pth"
        "baseRGB_Shiftnet_T01_seed13.pth"
        "baseRGB_Shiftnet_T01_seed17.pth"
        "baseRGB_Shiftnet_T03_seed10.pth"
        "baseRGB_Shiftnet_T03_seed13.pth"
        "baseRGB_Shiftnet_T03_seed17.pth"
        "baseRGB_Shiftnet_T05_seed10.pth"
        "baseRGB_Shiftnet_T05_seed13.pth"
        "baseRGB_Shiftnet_T05_seed17.pth"
        "baseRGB_Shiftnet_T07_seed10.pth"
        "baseRGB_Shiftnet_T07_seed13.pth"
        "baseRGB_Shiftnet_T07_seed17.pth"
        "baseRGB_Shiftnet_T09_seed10.pth"
        "baseRGB_Shiftnet_T09_seed13.pth"
        "baseRGB_Shiftnet_T09_seed17.pth"
        "baseRGB_Shiftnet_T11_seed10.pth"
        "baseRGB_Shiftnet_T11_seed13.pth"
        "baseRGB_Shiftnet_T11_seed17.pth"
    )
    # Download the files
    for file in "${davide_ckpt_files[@]}"; do
        curl -u "${USERNAME}:${PASSWORD}" -O "${davide_ckpt_root}/${file}"
    done
    # Go back to the checkpoint directory
    cd ..
fi

if [ "$EXP" = "E2" ] || [ "$EXP" = "all" ]; then
    # Create a directory for E2 checkpoints and switch to it
    mkdir -p "E2_SOTA_comparison"
    cd "E2_SOTA_comparison"
    # Set the root URL for the DAVIDE checkpoints
    davide_ckpt_root="https://davide.rd.tuni.fi/checkpoints/E2_SOTA_comparison"
    # List of files to download:
    davide_ckpt_files=(
        "DGN_multi_frame.pth"
        "DGN_single_frame.pth"
        "EDVR.pth"
        "RVRT.pth"
        "VRT.pth"
    )
    # Download the files
    for file in "${davide_ckpt_files[@]}"; do
        curl -u "${USERNAME}:${PASSWORD}" -O "${davide_ckpt_root}/${file}"
    done

    # Set checkpoints for ShiftNet variants
    #
    # check if file exists: ../E1_depth_impact/RGBD_Shiftnet_T11_seed13.pth
    if [ -f "../E1_depth_impact/RGBD_Shiftnet_T11_seed13.pth" ]; then
        # if the file exists, link it
        #
        # link the file 
        ln -s "../E1_depth_impact/RGBD_Shiftnet_T11_seed13.pth" "./Shiftnet_RGBD.pth"
    else
        # if the file does not exist, download it
        #
        # Set the root URL for the DAVIDE checkpoints
        davide_ckpt_root="https://davide.rd.tuni.fi/checkpoints/E1_depth_impact"
        file="RGBD_Shiftnet_T11_seed13.pth"
        curl -u "${USERNAME}:${PASSWORD}" -O "${davide_ckpt_root}/${file}"
        # rename the file
        mv "${file}" "Shiftnet_RGBD.pth"
    fi
    # check if file exists: ../E1_depth_impact/baseRGB_Shiftnet_T11_seed10.pth
    if [ -f "../E1_depth_impact/baseRGB_Shiftnet_T11_seed10.pth" ]; then
        # if the file exists, link it
        #
        # link the file 
        ln -s "../E1_depth_impact/baseRGB_Shiftnet_T11_seed10.pth" "./Shiftnet_baseRGB.pth"
    else
        # if the file does not exist, download it
        #
        # Set the root URL for the DAVIDE checkpoints
        davide_ckpt_root="https://davide.rd.tuni.fi/checkpoints/E1_depth_impact"
        file="baseRGB_Shiftnet_T11_seed10.pth"
        curl -u "${USERNAME}:${PASSWORD}" -O "${davide_ckpt_root}/${file}"
        # rename the file
        mv "${file}" "Shiftnet_baseRGB.pth"
    fi

    # Go back to the checkpoint directory
    cd ..
fi

# Download License file
curl -u "${USERNAME}:${PASSWORD}" -O "https://davide.rd.tuni.fi/checkpoints/DAVIDE%20model%20weigths-ResearchRAIL.txt"