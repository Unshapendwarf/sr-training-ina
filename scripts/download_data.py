import os, zipfile, gdown

url = "https://drive.google.com/file/d/1Y0-82niwFXYeW8JNcUE-6I-9KB3gEehw/view?usp=share_link"

# make sure that you are in the root of working directory
out_path = os.path.join("./data.zip")
if not os.path.exists(out_path):
    gdown.download(url, output=out_path, quiet=False, fuzzy=True)
else:
    print(f"already exists: {out_path}")

data_zip = zipfile.ZipFile(out_path)
unzip_path = os.path.join("./data")

if not os.path.exists(unzip_path):
    data_zip.extractall(unzip_path)
else:
    print(f"already exists: {unzip_path}")
data_zip.close()
