import os, zipfile, gdown

url = 'https://drive.google.com/uc?id=1ussHhGVh0BEe_RjyGgD3lS3rJNwtOc4R'
out_path = os.path.join("../data.zip")
if not os.path.exists(out_path):
    gdown.download(url, out_path, quiet=False)
else:
    print(f"already exists: {out_path}")

data_zip = zipfile.ZipFile(out_path)
unzip_path = os.path.join("../data")

if not os.path.exists(unzip_path):
    data_zip.extractall(unzip_path)
else:
    print(f"already exists: {unzip_path}")
data_zip.close()
