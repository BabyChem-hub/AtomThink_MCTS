import os
import json
from loguru import logger
from os.path import join, isdir, isfile
import glob # 导入 glob 来查找文件

# --- 配置 ---
NUM_GPUS = 6 # 设置为您使用的 GPU 数量 (您是 6 个)
NODE_RANK = 0 # 假设节点编号是 0
BASE_OUTPUT_DIR = "evaluation_results/MathVerse_mcts" # 您的主输出目录
DATASET_NAME = "MathVerse" # 数据集名称
# 您运行推理时使用的 tag (这决定了错误目录的名字)
# 根据您的 ls 结果，这个 tag 就是 "mcts_slow"
EXPECTED_TAG = "mcts_slow"
# 为最终合并的文件起一个新名字
MERGED_FILENAME = f"answers_{EXPECTED_TAG}_merged_manual.json"
# --- 结束配置 ---

def read_json(path):
    try:
        # 尝试用 utf-8-sig 读取以处理可能的 BOM (Byte Order Mark)
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"读取 JSON 文件失败 {path}: {e}")
        return None

def save_json(filename, ds):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文等字符正确写入
            json.dump(ds, f, indent=4, ensure_ascii=False)
        logger.info(f"已将合并后的数据保存到 {filename}")
    except Exception as e:
        logger.error(f"保存 JSON 文件失败 {filename}: {e}")

def main():
    # 指向包含 0_0_0, 0_1_0 等子目录的上级目录
    base_inference_dir = join(BASE_OUTPUT_DIR, 'inference', DATASET_NAME)
    merged_data_list = [] # 假设结果是列表形式

    logger.info(f"开始手动合并 tag 为 '{EXPECTED_TAG}' 的结果...")

    for gpu_id in range(NUM_GPUS):
        task_id = 0 # 假设 tasks_per_gpu = 1

        # 构建指向【名字错误的目录】的路径
        # 它被命名为像 answers_mcts_slow_0_0_0.json 这样的目录
        incorrect_dir_name = f"answers_{EXPECTED_TAG}_{NODE_RANK}_{gpu_id}_{task_id}.json"
        # 完整的错误目录路径应该是 base_inference_dir/0_0_0/answers_...json
        incorrect_dir_path = join(base_inference_dir, f"{NODE_RANK}_{gpu_id}_{task_id}", incorrect_dir_name)

        if not isdir(incorrect_dir_path):
            logger.warning(f"未找到名字错误的目录，跳过: {incorrect_dir_path}")
            continue

        # 在这个错误的目录里查找【真正的 JSON 文件】
        # 根据您的 ls 结果，文件名似乎比较固定，但我们用通配符更保险
        # 查找目录下所有 .json 文件
        found_files = glob.glob(join(incorrect_dir_path, "*.json"))

        if not found_files:
             logger.warning(f"在目录 {incorrect_dir_path} 中未找到任何 .json 文件。")
             continue
        elif len(found_files) > 1:
             logger.warning(f"在目录 {incorrect_dir_path} 中找到多个 .json 文件，只处理第一个: {found_files[0]}")
             # 如果您确定只有一个是结果文件，可以保留这个逻辑
             # 否则您可能需要更精确的文件名匹配

        actual_json_path = found_files[0] # 使用找到的第一个 json 文件
        logger.info(f"找到实际的 JSON 文件: {actual_json_path}")
        data = read_json(actual_json_path)

        if data is not None:
            # 假设所有 GPU 生成的数据都是列表形式，将它们合并
            if isinstance(data, list):
                merged_data_list.extend(data)
            # 如果数据可能是字典形式，您需要调整合并逻辑
            # elif isinstance(data, dict):
            #     merged_data_dict.update(data) # 如果是字典用 update
            else:
                 logger.warning(f"文件 {actual_json_path} 中的数据既不是列表也不是字典，无法合并。类型: {type(data)}")


    # 确定最终保存的文件路径 (保存在 base_inference_dir 下)
    final_merged_path = join(base_inference_dir, MERGED_FILENAME)

    # 保存合并后的列表
    if merged_data_list:
        save_json(final_merged_path, merged_data_list)
        logger.info(f"合并完成！总共合并了 {len(merged_data_list)} 个样本。")
    # 如果您需要处理字典形式的结果，在这里添加保存字典的逻辑
    # elif merged_data_dict:
    #     save_json(final_merged_path, merged_data_dict)
    #     logger.info(f"合并完成！总共合并了 {len(merged_data_dict)} 个样本。")
    else:
        logger.warning("没有找到任何有效数据进行合并。")

if __name__ == "__main__":
    main()