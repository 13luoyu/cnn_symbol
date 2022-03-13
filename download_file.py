import os
import requests
import zipfile
import tarfile

def download(name, cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    filename=os.path.join(cache_dir, name.split('/')[-1])
    if os.path.exists(filename):
        return filename
    print(f'Download {filename} from {name}...')
    r = requests.get(name, stream=True, verify=True)
    with open(filename, 'wb') as f:
        f.write(r.content)
    return filename

def download_and_extract(name):
    filename=download(name)
    dirname=os.path.dirname(filename)
    file, ext = os.path.splitext(filename)
    if os.path.exists(file):
        return file
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(filename, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(dirname)
    return file