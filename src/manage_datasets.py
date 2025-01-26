import os
import subprocess

def track_dataset(version_message):
    # Add dataset to DVC
    subprocess.run(["dvc", "add", "data/raw_data.csv"], check=True)
    
    # Add changes to Git
    subprocess.run(["git", "add", "data/raw_data.csv.dvc", ".gitignore"], check=True)
    
    # Commit changes with a version message
    subprocess.run(["git", "commit", "-m", version_message], check=True)

if __name__ == "__main__":
    # Example: Track a new version of the dataset
    track_dataset("Update dataset to version 2")
