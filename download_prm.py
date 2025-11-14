# download_prm.py
from huggingface_hub import snapshot_download

PRM_REPO_ID = "Qwen/Qwen2.5-Math-PRM-7B" # 替换为正确的 Hub 名称
DOWNLOAD_PATH = "/data1/xiangkun/MODELS/Qwen2.5-Math-PRM-7B-local" 

print(f"开始下载 PRM 模型到 {DOWNLOAD_PATH}...")
snapshot_download(
    repo_id=PRM_REPO_ID,
    local_dir=DOWNLOAD_PATH,
    local_dir_use_symlinks=False,
)
print("PRM 模型下载完成！")