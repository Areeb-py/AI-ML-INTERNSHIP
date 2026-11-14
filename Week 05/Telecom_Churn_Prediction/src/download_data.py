"""Download the IBM Telco Customer Churn CSV into data/raw/."""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Telco Customer Churn dataset...")
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    if len(content) < 10_000:
        raise RuntimeError("Downloaded file is unexpectedly small; check the source URL.")
    DESTINATION.write_bytes(content)
    print(f"Saved {len(content):,} bytes to: {DESTINATION}")

if __name__ == "__main__":
    main()
