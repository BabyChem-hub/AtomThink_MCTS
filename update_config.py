# 文件: update_config.py
import os
from huggingface_hub import hf_hub_download
import shutil

# ----------------------------------------------------------------------
# 📌 替换为你的模型信息
# ----------------------------------------------------------------------
REPO_ID = "liuhaotian/llava-v1.5-7b" 
LOCAL_MODEL_PATH = "/data1/xiangkun/MODELS/Llama-3.2-Vision-11B" 
CONFIG_FILENAME = "config.json"
# ----------------------------------------------------------------------

print(f"正在从 Hugging Face Hub 下载 {REPO_ID} 的 {CONFIG_FILENAME}...")

try:
    # 1. 下载最新的 config.json 到一个临时缓存位置
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=CONFIG_FILENAME,
        # 强制忽略本地缓存，从 Hub 下载最新版本
        force_download=True 
    )

    # 2. 将下载的文件复制到你的本地模型目录，覆盖旧文件
    destination_path = os.path.join(LOCAL_MODEL_PATH, CONFIG_FILENAME)
    shutil.copy(local_path, destination_path)

    print(f"✅ {CONFIG_FILENAME} 已成功下载并更新到: {destination_path}")

except Exception as e:
    print(f"❌ 配置文件更新失败。请检查 REPO_ID 和网络连接。错误信息: {e}")