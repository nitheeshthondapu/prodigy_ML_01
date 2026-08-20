import os
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Configure custom user-agent to avoid potential bot blocks
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response:
        with open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
    print("Download completed successfully.")

def main():
    base_url = "https://raw.githubusercontent.com/sidhantagar/Kaggle-House-Prices/master/"
    files = ["train.csv", "test.csv", "data_description.txt"]
    
    # Resolve project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(os.path.dirname(script_dir), "data", "raw")
    
    for filename in files:
        url = base_url + filename
        dest = os.path.join(raw_dir, filename)
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    main()
