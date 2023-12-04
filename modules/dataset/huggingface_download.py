import requests
import os
from tqdm import tqdm
 
urls = [
    "https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/pytorch_model.bin"
]
 
filepath = "WizardCoder/WizardCoder-15B-V1.0"
 
 
def download_file(url):
    filename = url.split("/")[-1]
    download_path = os.path.join(filepath, filename)
 
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()
 
    file_size = int(response.headers.get("Content-Length", 0))  # 获取待下载的文件大小
    chunk_size = 8192  # 读取的数据块的大小是8千字节
    
    with open(download_path, "wb") as file, tqdm(
        total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=filename
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(1)
 
 
for url in urls:
    download_file(url)