import os
import gdown
import zipfile
from pathlib import Path

def get_project_root():
    """Get the path to the project root directory"""
    current_file = Path(__file__).resolve()
    return str(current_file.parent.parent.parent)

def create_directory(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_and_extract(file_id, output_zip, extract_path):
    # Create directory if it doesn't exist
    create_directory(extract_path)
    
    # Download the file from Google Drive
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_zip, quiet=False)
    
    # Extract the zip file
    print(f"Extracting {output_zip} to {extract_path}")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Remove the zip file after extraction
    os.remove(output_zip)
    print(f"Removed {output_zip}")

def main():
    # File IDs from the Google Drive links
    ffhq_file_id = '1KUxJ-G6CBFzYpeg4PfTL93N8YybNExA7'
    celeba_hq_file_id = '1Fbqp_4zkNCj2SE1pcJUs4K3HS_yhFCqB'
    
    # Get project root path and set data directory
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    
    # Update paths with project root
    ffhq_zip = os.path.join(data_dir, 'ffhq_256.zip')
    celeba_zip = os.path.join(data_dir, 'celeba_hq_256.zip')
    ffhq_dir = os.path.join(data_dir, 'ffhq')
    celeba_dir = os.path.join(data_dir, 'celeba_hq')
    
    # Create data directory
    create_directory(data_dir)
    
    # Download and extract FFHQ dataset
    print("Downloading FFHQ dataset...")
    download_and_extract(ffhq_file_id, ffhq_zip, ffhq_dir)
    print("FFHQ dataset downloaded and extracted successfully!")
    
    # Download and extract CelebA-HQ dataset
    print("\nDownloading CelebA-HQ dataset...")
    download_and_extract(celeba_hq_file_id, celeba_zip, celeba_dir)
    print("CelebA-HQ dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
