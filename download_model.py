import requests
import zipfile
import os
import sys

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    print("\nDownload completed!")

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Download model
    model_url = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
    zip_path = "models/vosk-model-en-us-0.22.zip"
    download_file(model_url, zip_path)

    # Extract the zip file
    print("Extracting model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('models')
    
    # Remove the zip file
    os.remove(zip_path)
    print("Model downloaded and extracted successfully!")

if __name__ == "__main__":
    main() 