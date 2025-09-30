import zipfile
import os

def extract_dataset():
    try:
        with zipfile.ZipFile('bank-additional.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        # Rename the extracted file to match our expected filename
        os.rename('bank-additional/bank-additional-full.csv', 'bank-full.csv')
        print("Dataset extracted successfully!")
    except Exception as e:
        print(f"Error extracting dataset: {str(e)}")

if __name__ == "__main__":
    extract_dataset()