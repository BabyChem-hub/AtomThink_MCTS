# 文件: download_llama.py

import os
from huggingface_hub import snapshot_download

# -------------------------------------------------------------
# 📌 步骤 1: 替换为你需要下载的模型名称
# -------------------------------------------------------------
# 请使用 'llama32vision 11b' 模型在 Hub 上的正确名称
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct" 

# -------------------------------------------------------------
# 📌 步骤 2: 替换为你希望保存模型的绝对路径
# -------------------------------------------------------------
DOWNLOAD_PATH = "/data1/xiangkun/MODELS/llama32vision-11b-local" 
# -------------------------------------------------------------


# 创建目标文件夹（如果不存在）
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

print(f"开始下载 {MODEL_NAME} 到 {DOWNLOAD_PATH}...")

try:
    # 使用 snapshot_download 函数直接下载整个仓库内容
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=DOWNLOAD_PATH,
        local_dir_use_symlinks=False,  # 确保所有文件都是物理复制，而不是符号链接
    )
    print("模型下载完成！")
    print(f"模型路径已设置在: {DOWNLOAD_PATH}")

except Exception as e:
    print(f"下载失败。请检查模型名称是否正确，以及你是否有访问该模型的权限。错误信息: {e}")