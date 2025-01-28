from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
CSV_FILES = os.getenv("CSV_FILES", "").split(",")
REPO_ID = os.getenv("REPO_ID", "")

def validate_env_vars():
    if not HUGGINGFACE_TOKEN:
        raise ValueError("Hugging Face token not found in environment variables.")
    if not REPO_ID:
        raise ValueError("Repository ID not provided.")
    if not CSV_FILES:
        raise ValueError("No CSV files specified for upload.")

def upload_csv_to_hf(csv_files, repo_id, token):
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True, private=True)
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue
        print(f"Uploading {csv_file}...")
        api.upload_file(
            path_or_fileobj=csv_file,
            path_in_repo=os.path.basename(csv_file),
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"Successfully uploaded {csv_file}")


def load_hf_to_csv(repo_id, token, dest_path="../data/"):
    """
    Load CSV files from Hugging Face Hub
    
    Args:
        repo_id (str): Hugging Face repository ID (format: 'username/repo-name')
        token (str): Hugging Face API token
        dest_path (str): Local directory to save downloaded files
    """
    api = HfApi()
    os.makedirs(dest_path, exist_ok=True)
    
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        for file in files:
            if file.endswith(".csv"):
                print(f"Downloading {file}...")
                from huggingface_hub import hf_hub_download
                
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    repo_type="dataset",
                    token=token,
                    local_dir=dest_path
                )
                print(f"Successfully downloaded {file} to {dest_path}")
    except Exception as e:
        print(f"Error downloading files: {str(e)}")


if __name__ == "__main__":
    try:
        validate_env_vars()
        upload_csv_to_hf(CSV_FILES, REPO_ID, HUGGINGFACE_TOKEN)
        load_hf_to_csv(REPO_ID, HUGGINGFACE_TOKEN)
    except Exception as e:
        print(f"An error occurred: {e}")
