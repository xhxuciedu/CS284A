"""Download ProteinGym's official small supervised-fold archive once."""
from pathlib import Path
from zipfile import ZipFile

import requests

URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip"


def main():
    data = Path("data")
    data.mkdir(exist_ok=True)
    archive = data / "cv_folds_singles_substitutions.zip"
    if not archive.exists():
        with requests.get(URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (data / member.filename).resolve()
            if not destination.is_relative_to(data.resolve()):
                raise ValueError("Unsafe path in dataset archive")
        zipped.extractall(data)
    csvs = list(data.rglob("*.csv"))
    print(f"Extracted {len(csvs)} CSV files under {data.resolve()}")


if __name__ == "__main__":
    main()
