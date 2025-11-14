import json
import os
from tqdm import tqdm # 用于显示进度条，可以 pip install tqdm

# --- ⬇️ 你需要在这里修改变量 ⬇️ ---

# 1. SFT 数据集 (你的图片来源)
SFT_DATA_PATH = "path/to/your/AtomMATH-SFT.jsonl" 

# 2. PRM 数据集 (你需要被补充的数据集)
PRM_DATA_PATH = "path/to/your/PRM-dataset.jsonl"

# 3. 输出文件 (补充图片后的新数据集)
OUTPUT_DATA_PATH = "path/to/your/PRM-dataset_enriched.jsonl"

# 4. 两个文件里用来匹配的“共同字段”的名字
# (例如 'id', 'question', 'question_id' 等)
COMMON_KEY_FIELD = "id" 

# 5. 数据集中存放图片信息的字段名
IMAGE_FIELD_NAME = "image"

# --- ⬆️ 修改结束 ⬆️ ---


print("第一步: 正在从 SFT 数据集建立图片索引...")

sft_image_map = {}
try:
    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f_sft:
        for line in tqdm(f_sft):
            try:
                sft_entry = json.loads(line.strip())
                common_key = sft_entry.get(COMMON_KEY_FIELD)
                image_data = sft_entry.get(IMAGE_FIELD_NAME)
                
                if common_key is not None and image_data:
                    if common_key in sft_image_map:
                        print(f"警告: SFT 数据中 {COMMON_KEY_FIELD} '{common_key}' 出现重复，将使用后一个。")
                    sft_image_map[common_key] = image_data
                    
            except json.JSONDecodeError:
                print(f"警告: SFT 文件中有一行 JSON 格式错误，已跳过: {line[:50]}...")

except FileNotFoundError:
    print(f"错误: 找不到 SFT 文件: {SFT_DATA_PATH}")
    exit(1)

print(f"索引建立完毕，共加载 {len(sft_image_map)} 个带图片的条目。")
print("-" * 20)
print("第二步: 正在处理 PRM 数据集并合并图片...")

added_count = 0
missing_count = 0

try:
    with open(PRM_DATA_PATH, 'r', encoding='utf-8') as f_prm, \
         open(OUTPUT_DATA_PATH, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_prm):
            try:
                prm_entry = json.loads(line.strip())
                common_key = prm_entry.get(COMMON_KEY_FIELD)
                
                # 检查 PRM 条目是否已经有图片
                # .get(IMAGE_FIELD_NAME) 会处理字段不存在或值为 None/空字符串/空列表
                if not prm_entry.get(IMAGE_FIELD_NAME):
                    
                    # 没有图片，尝试从 SFT 索引中查找
                    if common_key in sft_image_map:
                        # 找到了！"粘贴"图片数据
                        prm_entry[IMAGE_FIELD_NAME] = sft_image_map[common_key]
                        added_count += 1
                    else:
                        # SFT 索引里也找不到，说明 SFT 里也没有
                        missing_count += 1
                
                # 将处理后 (或未处理) 的数据写入新文件
                f_out.write(json.dumps(prm_entry, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"警告: PRM 文件中有一行 JSON 格式错误，已跳过: {line[:50]}...")

except FileNotFoundError:
    print(f"错误: 找不到 PRM 文件: {PRM_DATA_PATH}")
    exit(1)

print("-" * 20)
print("处理完成！")
print(f"✅ 成功补充了 {added_count} 条数据的图片。")
print(f"⚠️ {missing_count} 条数据在 SFT 中也未找到对应图片。")
print(f"➡️ 已生成新文件: {OUTPUT_DATA_PATH}")