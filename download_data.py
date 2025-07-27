# pylint: disable-all

import os
import glob

# Set the directory containing kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath('./kaggle')
# Ensure the directory exists
if not os.path.exists(os.environ['KAGGLE_CONFIG_DIR']):
    raise FileNotFoundError(f"Kaggle config directory not found: {os.environ['KAGGLE_CONFIG_DIR']}")

from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate the Kaggle API
api = KaggleApi()
api.authenticate()

# List all files in the competition
files = api.competition_list_files('ariel-data-challenge-2025').files

# Download all files, preserving folder structure
for f in files:
    # Retain the same structure as on Kaggle
    file_path = os.path.join('data', f.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Download the file
    print(f"Downloading {f.name}...")
    api.competition_download_file('ariel-data-challenge-2025', f.name, path=os.path.dirname(file_path))
    print(f"Downloaded {f.name} to {file_path}")

# Use glob to index all downloaded files and subfolders
downloaded_files = glob.glob('data/**/*', recursive=True)
print("Downloaded files and folders:")
for file in downloaded_files:
    print(file)