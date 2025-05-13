from pathlib import Path
file_path = "~/Documents"
file_path = Path(file_path).expanduser()

print(file_path.parent)
print(file_path)  # /home/user/Documents/file.txt