import os
import urllib.request
import tarfile

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"

if not os.path.exists("aclImdb"):
    print("📦 正在下载 IMDB 数据集...")
    urllib.request.urlretrieve(url, filename)

    print("📂 正在解压...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    os.remove(filename)

print("✅ IMDB 数据准备完成！")
