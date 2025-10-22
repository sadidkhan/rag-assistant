import os, json, uuid
from fastapi import UploadFile

UPLOAD_DIR = "data/uploads/"
META_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class UploadFileService:
    def __init__(self):
        if not os.path.exists(META_FILE):
            with open(META_FILE, "w") as f:
                json.dump([], f)

    def _load_meta(self):
        with open(META_FILE, "r") as f:
            return json.load(f)

    def _save_meta(self, data):
        with open(META_FILE, "w") as f:
            json.dump(data, f, indent=4)

    async def save_upload(self, file: UploadFile):
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        if os.path.exists(file_path):
            raise FileExistsError(f"File already exists: {file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        entry = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "filepath": file_path,
            "isIndexed": False
        }

        data = self._load_meta()
        data.append(entry)
        self._save_meta(data)

        return entry
    
    async def get_file_names(self):
        return self._load_meta()


    def mark_indexed(self, file_id: str, flag: bool = True):
        data = self._load_meta()
        for item in data:
            if item["id"] == file_id:
                item["isIndexed"] = flag
                break
        self._save_meta(data)
