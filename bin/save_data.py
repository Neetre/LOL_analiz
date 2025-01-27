from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
CSV_FILES = os.getenv("CSV_FILES").split(",")
REPO_ID = os.getenv("REPO_ID")

def upload_csv_to_hf(csv_files, repo_id, token):
    """
    Upload CSV files to Hugging Face Hub
    
    Args:
        csv_files (list): List of paths to CSV files
        repo_id (str): Hugging Face repository ID (format: 'username/repo-name')
        token (str): Hugging Face API token
    """
    api = HfApi()
    
    try:
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
            
    except Exception as e:
        print(f"Error uploading files: {str(e)}")


def load_hf_to_csv(repo_id, token):
    """
    Load CSV files from Hugging Face Hub
    
    Args:
        repo_id (str): Hugging Face repository ID (format: 'username/repo-name')
        token (str): Hugging Face API token
    """
    api = HfApi()
    
    try:
        files = api.list_files(repo_id=repo_id, token=token)
        for file in files:
            if file["filename"].endswith(".csv"):
                print(f"Downloading {file['filename']}...")
                api.download_file(
                    repo_id=repo_id,
                    filename=file["filename"],
                    path="../data/",
                    token=token
                )
                print(f"Successfully downloaded {file['filename']}")
                
    except Exception as e:
        print(f"Error downloading files: {str(e)}")


if __name__ == "__main__":
    upload_csv_to_hf(CSV_FILES, REPO_ID, HUGGINGFACE_TOKEN)
