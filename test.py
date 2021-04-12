from access.utils.paths import CHECKPOINT_DIR, REPO_DIR
from pathlib import Path
import os

print(os.path.abspath(Path("./bpe/encoder.json").parent))

print(Path( REPO_DIR / 'bpe'))

print(str(os.path.abspath(Path("./bpe/encoder.json").parent)) != str(Path( REPO_DIR / 'bpe')))