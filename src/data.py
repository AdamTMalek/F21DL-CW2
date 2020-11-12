import os
from pathlib import Path


def get_data_dir_path() -> str:
    project_dir = Path(__file__).parent.parent.absolute()
    data_dir = os.path.join(project_dir, 'data')
    return data_dir
