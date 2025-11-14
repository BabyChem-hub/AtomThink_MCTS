from huggingface_hub import snapshot_download

# 定义你要下载的模型
repo_id = "liuhaotian/llava-v1.5-7b"

# 定义你想把模型保存在哪里的本地路径
local_path = "/data1/xiangkun/MODELS/llava-v1.5-7b-liu"  # 你可以改成你想要的任何路径

print(f"开始下载模型 {repo_id} 到 {local_path}...")

# 执行下载
snapshot_download(
    repo_id=repo_id,
    local_dir=local_path,
    local_dir_use_symlinks=False  # 建议设为 False，避免 Windows 上的符号链接问题
)

print("模型下载完成！")