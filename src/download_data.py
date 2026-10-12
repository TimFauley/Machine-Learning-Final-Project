"""Download and unzip the UCI HAR dataset into data/.

Usage: python src/download_data.py
"""
from pathlib import Path
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    outer = DATA / "har.zip"
    if not outer.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, outer)
    with zipfile.ZipFile(outer) as z:
        z.extractall(DATA)
    # The UCI archive wraps the real dataset in a second zip.
    inner = DATA / "UCI HAR Dataset.zip"
    if inner.exists():
        with zipfile.ZipFile(inner) as z:
            z.extractall(DATA)
    print(f"Done. Dataset at: {DATA / 'UCI HAR Dataset'}")


if __name__ == "__main__":
    main()
