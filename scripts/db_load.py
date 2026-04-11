

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from pathlib import Path
import kagglehub


download_path = kagglehub.dataset_download("ruchi798/data-science-job-salaries")

BASE_DIR = Path(__file__).resolve().parents[1]  
target_path = BASE_DIR / "data"

target_path.mkdir(parents=True, exist_ok=True)


for item in Path(download_path).iterdir():
    dest = target_path / item.name

    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)

print(f"Dataset copied to: {target_path}")