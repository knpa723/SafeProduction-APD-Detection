import os
import shutil
import gdown
import hashlib

def download_files(url: str, destination: str = "src/models"):
    temp_directory = os.path.join(destination, "__temp_gdown__")
    
    gdown.download_folder(
        url,
        output=temp_directory,
        quiet=False,
        use_cookies=False
    )
    
    for root, _, files in os.walk(temp_directory):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(destination, file)
            
            if os.path.exists(dst):
                if sha256(src) == sha256(dst):
                    print("Models still the same")
                    os.remove(src)
                    continue
            
            print(f"Updating: {file}")
            shutil.move(src, dst)
    
    shutil.rmtree(temp_directory)
    
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()