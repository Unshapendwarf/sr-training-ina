import os, zipfile, gdown

url = "https://drive.google.com/file/d/1FbhQgGcK6quIl5lgx9v0E9Eh3Yu64APP/view?usp=drive_link"

# make sure that you are in the root of working directory
out_path = os.path.join("./result_model.zip")
if not os.path.exists(out_path):
    gdown.download(url, output=out_path, quiet=False, fuzzy=True)
else:
    print(f"already exists: {out_path}")

data_zip = zipfile.ZipFile(out_path)
unzip_path = os.path.join("./result_model")

if not os.path.exists(unzip_path):
    data_zip.extractall(unzip_path)
else:
    print(f"already exists: {unzip_path}")
data_zip.close()
