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

output = 'data'  # Directory to save downloaded files
if not os.path.exists(output):
    os.makedirs(output)

# Fetch all files using pagination
all_files = []


page_size = 200  # Adjust page size as needed

files = api.competition_list_files('ariel-data-challenge-2025', page_size=page_size).files

for file in files[:100]: # Adjust the range as needed
    file_path = os.path.join(output, file.name)
    if not os.path.exists(file_path):
        print(f"Downloading {file.name}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        api.competition_download_file('ariel-data-challenge-2025', file.name, path=file_path)
    else:
        print(f"File {file.name} already exists. Skipping download.")
        
        
