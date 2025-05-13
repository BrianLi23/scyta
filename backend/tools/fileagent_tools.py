import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import hashlib
import mimetypes
from datetime import datetime
import pwd
import concurrent.futures  

class FileAgentTools:
    def __init__(self):
        self.mimetypes = mimetypes
        self.mimetypes.init()

    def get_metadata(self, file_path: Union[str, List[str]], metadata_cache: Dict = None) -> Union[Dict, List[Dict]]:
        """
        Return metadata of a file/folder given its path, process multiple files if given a list
        """
        if isinstance(file_path, list):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(lambda path: self._get_metadata_single(path, metadata_cache), file_path)
                return list(results)
        return self._get_metadata_single(file_path, metadata_cache)

    def _get_metadata_single(self, file_path: str, metadata_cache: Dict = None) -> Dict:
        """
        Get metadata of a file/folder given pathname
        """
        try:
            file_path = Path(file_path).expanduser()
            # If file path is in cache, return it
            if metadata_cache and str(file_path) in metadata_cache:
                return metadata_cache[str(file_path)]

            stat_info = file_path.stat()
            file_hash = self._get_file_hash(file_path) if file_path.is_file() else None
            file_type, encoding = mimetypes.guess_type(str(file_path))

            metadata = {
                "file_path": str(file_path),
                "file_owner": pwd.getpwuid(stat_info.st_uid).pw_name,
                "size": stat_info.st_size,
                "parent_directory": str(file_path.parent),
                "type": "directory" if file_path.is_dir() else "file",
                "timestamps": {
                    "last_modified": datetime.fromtimestamp(stat_info.st_mtime),
                    "last_accessed": datetime.fromtimestamp(stat_info.st_atime),
                    "created": datetime.fromtimestamp(stat_info.st_birthtime)
                },
                "permissions": {
                    "mode": oct(stat_info.st_mode)[-3:],
                    "readable": os.access(file_path, os.R_OK),
                    "writable": os.access(file_path, os.W_OK),
                    "executable": os.access(file_path, os.X_OK)
                }
            }
            
            if metadata["type"] == "file":
                metadata.update({
                    "file_hash": file_hash,
                    "mime_type": file_type,
                    "extension": file_path.suffix,
                    "size_formatted": self._format_size(stat_info.st_size)
                })
            else:
                try:
                    contents = list(file_path.iterdir())
                    metadata.update({
                        "contents_count": {
                            "files": sum(1 for x in contents if x.is_file()),
                            "directories": sum(1 for x in contents if x.is_dir())
                        }
                    })
                except PermissionError:
                    metadata["error"] = "Permission denied to read directory contents"

            if metadata_cache is not None:
                metadata_cache[str(file_path)] = metadata
            return metadata

        except Exception as e:
            return {"error": str(e)}

    def _format_size(self, size: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"

    def _get_file_hash(self, file_path):
        """Return the SHA-256 hash of the file at the given path."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def create_file(self, file_path: str, content: str) -> Dict:
        """Create a new file with given content"""
        try:
            file_path = Path(file_path).expanduser()
            with open(file_path, "w") as file:
                file.write(content)
            return {"message": "File created successfully"}
        except Exception as e:
            return {"error": str(e)}

    def create_dir(self, file_path: str) -> Dict:
        """Create a new directory"""
        try:
            true_path = Path(file_path).expanduser()
            true_path.mkdir(parents=True, exist_ok=True)
            return {"message": "Directory created successfully"}
        except Exception as e:
            return {"error": str(e)}

    def delete_file(self, file_path: Union[str, List[str]]) -> Dict:
        """Move file(s) to trash"""
        try:
            trash_path = Path.home() / ".Trash"
            results = []
            
            if isinstance(file_path, list):
                for path in file_path:
                    path = Path(path).expanduser()
                    if not path.exists():
                        results.append({"file": str(path), "error": "File does not exist"})
                        continue
                    shutil.move(path, trash_path / path.name)
                    results.append({"file": str(path), "message": "File moved to trash successfully"})
                return {"results": results}
            else:
                path = Path(file_path).expanduser()
                if not path.exists():
                    return {"error": "File does not exist"}
                shutil.move(path, trash_path / path.name)
                return {"message": "File moved to trash successfully"}
        except Exception as e:
            return {"error": str(e)}

    def move_file(self, source_path: Union[str, List[str]], destination_path: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Move file(s) from source to destination"""
        try:
            if isinstance(source_path, list):
                if not isinstance(destination_path, list):
                    dest_path = Path(destination_path).expanduser()
                    if not dest_path.is_dir():
                        return {"error": "Destination path must be a directory"}
                    dest_paths = [dest_path / Path(src).name for src in source_path]
                    return [self._move_file_single(src, dest) for src, dest in zip(source_path, dest_paths)]
                
                if len(source_path) != len(destination_path):
                    return {"error": "Source and destination paths must be of equal length"}
                return [self._move_file_single(src, dest) for src, dest in zip(source_path, destination_path)]
            
            return self._move_file_single(source_path, destination_path)
        except Exception as e:
            return {"error": str(e)}

    def _move_file_single(self, source_path: str, destination_path: str) -> Dict:
        """Move a single file from source to destination"""
        try:
            source_path = Path(source_path).expanduser()
            destination_path = Path(destination_path).expanduser()
            
            if not source_path.exists():
                return {"error": "File does not exist"}
            
            shutil.move(source_path, destination_path)
            return {"message": "File moved successfully"}
        except Exception as e:
            return {"error": str(e)}

    def copy_file(self, source_path: Union[str, List[str]], destination_path: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Copy file(s) from source to destination"""
        try:
            if isinstance(source_path, list):
                if not isinstance(destination_path, list):
                    dest_path = Path(destination_path).expanduser()
                    if not dest_path.is_dir():
                        return {"error": "Destination must be a directory when copying multiple files"}
                    dest_paths = [dest_path / Path(src).name for src in source_path]
                    return [self._copy_single_file(src, dest) for src, dest in zip(source_path, dest_paths)]
                
                if len(source_path) != len(destination_path):
                    return {"error": "Source and destination paths must be of equal length"}
                return [self._copy_single_file(src, dest) for src, dest in zip(source_path, destination_path)]
            return self._copy_single_file(source_path, destination_path)
        except Exception as e:
            return {"error": str(e)}

    def _copy_single_file(self, source_path: str, destination_path: str) -> Dict:
        """Copy a single file from source to destination"""
        try:
            source_path = Path(source_path).expanduser()
            destination_path = Path(destination_path).expanduser()
            
            if not os.path.exists(source_path):
                return {"error": "File does not exist"}
            shutil.copy(source_path, destination_path)
            return {"message": "File copied successfully"}
        except Exception as e:
            return {"error": str(e)}

    def rename_file(self, source_path: Union[str, List[str]], destination_path: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Rename file(s) from source to destination"""
        try:
            if isinstance(source_path, list):
                if not isinstance(destination_path, list):
                    return {"error": "Source and destination paths must be lists"}
                if len(source_path) != len(destination_path):
                    return {"error": "Source and destination paths must be of equal length"}
                return [self._rename_file_single(src, dest) for src, dest in zip(source_path, destination_path)]
            return self._rename_file_single(source_path, destination_path)
        except Exception as e:
            return {"error": str(e)}

    def _rename_file_single(self, source_path: str, destination_path: str) -> Dict:
        """Rename a single file from source to destination"""
        try:
            source_path = Path(source_path).expanduser()
            destination_path = Path(destination_path).expanduser()
            
            if not source_path.exists():
                return {"error": "File does not exist"}
            
            source_path.rename(destination_path)
            return {"message": "File renamed successfully"}
        except Exception as e:
            return {"error": str(e)}
        