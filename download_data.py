import urllib.request
import gzip
import shutil
import os

def download_file(url, filename):
    urllib.request.urlretrieve(url, filename)

def extract_gz(gz_path, out_path):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def main():
    os.makedirs('data', exist_ok=True)

    # Download and extract reviews
    reviews_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz'
    reviews_gz = 'data/reviews_Electronics_5.json.gz'
    reviews_json = 'data/reviews_Electronics_5.json'
    download_file(reviews_url, reviews_gz)
    extract_gz(reviews_gz, reviews_json)

    # Download and extract metadata
    meta_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz'
    meta_gz = 'data/meta_Electronics.json.gz'
    meta_json = 'data/meta_Electronics.json'
    download_file(meta_url, meta_gz)
    extract_gz(meta_gz, meta_json)

    print("Data downloaded and extracted successfully.")

if __name__ == "__main__":
    main()