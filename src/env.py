from pathlib import Path
from typing import Final

PACKAGE_DIR: Final[Path] = Path(__file__).parents[2]
RESULTS_DIR: Final[Path] = PACKAGE_DIR / 'data/results'
