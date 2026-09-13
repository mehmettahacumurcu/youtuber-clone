"""Keep source files that datasets hashes dynamically during import."""
from PyInstaller.utils.hooks import collect_data_files


datas = collect_data_files("datasets.packaged_modules", include_py_files=True)
