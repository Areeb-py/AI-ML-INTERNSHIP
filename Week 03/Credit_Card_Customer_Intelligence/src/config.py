
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "CC GENERAL.csv"
OUTPUT = ROOT / "outputs"
PLOTS = OUTPUT / "visualizations"
RANDOM_STATE = 42
CONTAMINATION = 0.02
