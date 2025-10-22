import os

def get_upload_path(filename: str) -> str:
    """Return the full path for an uploaded file."""
    base_dir = os.path.dirname(__file__)
    upload_dir = os.path.join(base_dir, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    return os.path.join(upload_dir, filename)
