import math
import subprocess
import argparse
import os
from loguru import logger
from os.path import join, dirname, abspath, isdir, isfile
from os import listdir, mkdir, makedirs
import sys
import random
import copy # 确保导入 copy 模块

# 使用 sys.executable 获取当前 Python 解释器路径
CURRENT_PYTHON_PATH = sys.executable

current_dir = dirname(os.path.abspath(__file__))
# 添加必要的项目路径到 sys.path (根据您的项目结构调整)
project_root = dirname(dirname(dirname(current_dir)))
src_root = dirname(dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
if src_root not in sys.path:
    sys.path.append(src_root)

print("sys.path:", sys.path)
print("Current Working Directory:", os.getcwd())

# 尝试导入，增加错误处理
try:
    from llamafactory.evaluation.utils.eval_utils import read_json, save_json, save_jsonl, read_jsonl, load_yaml, save_yaml
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您的 Python 环境配置正确，并且 llamafactory 包已正确安装或在 sys.path 中。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_rank', type=int, default=0, help="节点排名 (多节点推理时使用)")
    parser.add_argument('--total_gpus', type=int, default=None, help="所有节点的 GPU 总数；默认使用当前节点 GPU 数")
    parser.add_argument('--nproc_per_node', type=int, default=8, help="每个节点使用的 GPU 数量 (请根据实际情况修改!)")
    parser.add_argument('--tasks_per_gpu', type=int, default=1, help="每个 GPU 运行的任务数")
    parser.add_argument('--config', type=str, required=True, help="主模型配置文件路径 (必需)") # 设为必需
    parser.add_argument('--output_dir', type=str, default=None, help="覆盖配置文件中的 output_dir")
    parser.add_argument('--remote_work_dir', type=str, default=None, help="覆盖配置文件中的 remote_work_dir")
    parser.add_argument('--muti_gpu_per_task', action='store_true', help="是否每个任务使用多个 GPU (例如策略模型和奖励模型分开)")
    parser.add_argument('--gpus_per_task', type=int, default=None, help="每个任务占用的 GPU 数量；若未指定则根据 --muti_gpu_per_task 推断")
    parser.add_argument('--task', type=str, required=True, help="要评估的任务名称 (例如 MathVerse) (必需)") # 设为必需
    parser.add_argument('--prompt', type=str, default="base", help="使用的提示词模板名称")
    parser.add_argument('--method', type=str, default="base", choices=["base", "quick", "slow", "mcts"], help="使用的推理方法")
    parser.add_argument('--separate_eval', dest='separate_eval', action='store_true', help="是否进行独立评估 (通常为 True)")
    parser.add_argument('--no_separate_eval', dest='separate_eval', action='store_false', help="禁用独立评估")
    parser.add_argument('--max_sampling_count', type=int, default=300, help="AtomThink 中的最大采样次数")
    parser.add_argument('--max_samples', type=int, default=None, help="限制处理的总样本数量 (用于测试)")
    parser.add_argument('--temperature', type=float, default=0.0, help="生成温度")
    parser.add_argument('--atomthink_beam_search_num', type=int, default=2, help="AtomThink slow/quick 的束搜索宽度")
    parser.add_argument('--candidate_num', type=int, default=10, help="AtomThink slow/quick/mcts 的候选数量")
    parser.add_argument('--mcts_iterations', type=int, default=50, help="MCTS 每次搜索的迭代次数")
    parser.add_argument('--mcts_exploration_factor', type=float, default=1.41, help="MCTS 的 UCT 探索因子")
    parser.add_argument('--mcts_simulations', type=int, default=None, help="MCTS 模拟次数 (别名 for mcts_iterations)")
    parser.add_argument('--mcts_exploration_constant', type=float, default=None, help="MCTS 探索常数 (别名 for mcts_exploration_factor)")
    parser.add_argument(
        '--mcts_trace_samples',
        type=str,
        default=None,
        help="逗号/空格分隔的样本 index，用于输出 MCTS trace；使用 'all' 或 '*' 表示全部样本。",
    )
    parser.add_argument(
        '--mcts_trace_dir',
        type=str,
        default=None,
        help="覆盖配置中的 trace 输出目录；缺省时沿用 answers_file/mcts_traces。",
    )

    parser.set_defaults(separate_eval=True)
    args = parser.parse_args()
    
    # 处理参数别名映射
    if args.mcts_simulations is not None:
        args.mcts_iterations = args.mcts_simulations
        logger.info(f"使用 --mcts_simulations={args.mcts_simulations} 覆盖 mcts_iterations")
    
    if args.mcts_exploration_constant is not None:
        args.mcts_exploration_factor = args.mcts_exploration_constant
        logger.info(f"使用 --mcts_exploration_constant={args.mcts_exploration_constant} 覆盖 mcts_exploration_factor")

    # 兼容旧标志并确定实际的 gpus_per_task
    user_specified_gpus = args.gpus_per_task is not None
    if user_specified_gpus and args.gpus_per_task <= 0:
        parser.error("--gpus_per_task 必须是正整数")

    if not user_specified_gpus:
        if args.muti_gpu_per_task:
            args.gpus_per_task = 2
        else:
            auto_multi = args.method in ["mcts", "slow"] and args.nproc_per_node >= 2
            if auto_multi:
                logger.info("检测到 %s 方法，将自动启用双卡任务 (策略/奖励各占一张)。", args.method)
            args.gpus_per_task = 2 if auto_multi else 1
    else:
        if args.muti_gpu_per_task and args.gpus_per_task != 2:
            logger.warning("--muti_gpu_per_task 已启用，但 --gpus_per_task != 2；优先使用 --gpus_per_task 的值。")

    args.muti_gpu_per_task = args.gpus_per_task > 1
    logger.info(f"每个任务使用 GPU 数: {args.gpus_per_task}")
    
    return args


def detect_available_gpu_count():
    """检测当前可见的 GPU 数量，用于防止参数配置超过物理资源。"""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        device_list = [dev.strip() for dev in visible_devices.split(",") if dev.strip() != ""]
        # CUDA_VISIBLE_DEVICES 为 "-1" 表示禁用 GPU
        if device_list == ["-1"]:
            return 0
        if device_list:
            return len(device_list)
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return None

def get_task_config(task_name, base_config_dir="configs/inference"):
    """尝试加载指定任务的配置文件"""
    # 优先使用相对于当前脚本的路径构造
    script_dir = dirname(abspath(__file__))
    potential_path1 = join(script_dir, '../../..', base_config_dir, f"{task_name.lower()}.yaml")
    # 备用路径，相对于当前工作目录
    potential_path2 = join(base_config_dir, f"{task_name.lower()}.yaml")

    config_path = None
    if isfile(potential_path1):
        config_path = potential_path1
    elif isfile(potential_path2):
        config_path = potential_path2
    else:
        logger.error(f"任务配置文件未找到于: {potential_path1} 或 {potential_path2}")
        return None

    logger.info(f"加载任务配置文件: {config_path}")
    task_config = load_yaml(config_path)
    if task_config is None:
        logger.error(f"加载任务配置文件失败: {config_path}")
    return task_config


def parallel_dataset(args, task_config):
    """分割数据集文件到每个任务"""
    curPath = os.getcwd() # 假设脚本从项目根目录运行
    parallel_data_dir = join(curPath, "data", "test_data", "parallel", args.task)
    makedirs(parallel_data_dir, exist_ok=True)

    dataset_dir = task_config.get("dataset_dir")
    dataset_filename = task_config.get("dataset")
    if not dataset_dir or not dataset_filename:
        logger.error("任务配置文件缺少 'dataset_dir' 或 'dataset'")
        return False

    question_file = join(dataset_dir, dataset_filename)
    # 增加对绝对路径的支持
    if not isfile(question_file) and isfile(join(curPath, dataset_dir, dataset_filename)):
        question_file = join(curPath, dataset_dir, dataset_filename)
        logger.warning(f"使用相对项目根目录的路径: {question_file}")
    elif not isfile(question_file):
        # 【修正】 如果文件不存在，直接报错退出
        logger.error(f"数据集文件未找到: {question_file}")
        sys.exit(1) # 直接退出，而不是返回 False

    try:
        if question_file.endswith(".jsonl"):
            data = read_jsonl(question_file)
        elif question_file.endswith(".json"):
            data = read_json(question_file)
        else:
            logger.error(f"不支持的数据集文件格式: {question_file}")
            return False
    except Exception as e:
        logger.error(f"读取数据集文件失败 {question_file}: {e}")
        return False

    original_data_len = len(data) if isinstance(data, (list, dict)) else 0
    logger.info(f"原始数据集大小: {original_data_len}") # 增加日志

    # --- 【修改为随机抽样】 ---
    if args.max_samples is not None and args.max_samples > 0:
        max_samples_int = int(args.max_samples)
        if max_samples_int < original_data_len:
            logger.info(f"命令行 --max_samples ({max_samples_int}) 小于数据集大小 ({original_data_len})。将随机抽取 {max_samples_int} 个样本。")
            if isinstance(data, list):
                # 对列表随机抽样
                data = random.sample(data, max_samples_int)
            elif isinstance(data, dict):
                # 对字典的键进行随机抽样
                sampled_keys = random.sample(list(data.keys()), max_samples_int)
                data = {key: data[key] for key in sampled_keys}
            else:
                logger.warning(f"数据集类型 {type(data)} 不支持随机抽样。将使用所有 {original_data_len} 个样本。")
        else:
            logger.info(f"命令行 --max_samples ({max_samples_int}) 大于或等于数据集大小 ({original_data_len})。将使用所有样本。")
            # 如果 max_samples >= len(data)，则使用全部数据
    total_data = len(data) if isinstance(data, (list, dict)) else 0
    # --- 【随机抽样修改结束】 ---

    logger.info(f"加载并抽样后数据集: {question_file} (原始大小: {original_data_len}, 使用: {total_data})") # 修改日志

    # --- 基于全局任务数量计算分片 ---
    gpus_on_this_node = args.nproc_per_node
    tasks_per_node = gpus_on_this_node * args.tasks_per_gpu
    total_gpus = args.total_gpus if args.total_gpus else args.nproc_per_node
    total_tasks = total_gpus * args.tasks_per_gpu
    group_size = max(args.gpus_per_task, 1)
    effective_tasks_per_node = math.ceil(tasks_per_node / group_size)
    effective_total_tasks = math.ceil(total_tasks / group_size)

    if tasks_per_node <= 0 or total_tasks <= 0:
        logger.error(
            f"无效配置: tasks_per_node={tasks_per_node}, total_tasks={total_tasks}。"
            " 请检查 --nproc_per_node/--tasks_per_gpu/--total_gpus。"
        )
        return False

    if total_data == 0:
        logger.error(
            "抽样后数据集为空: %s。请检查 dataset_dir、dataset 以及 max_samples 配置。",
            question_file,
        )
        return False # 这里返回 False 让主流程停止

    data_per_task = math.ceil(total_data / effective_total_tasks)

    logger.info(
        "总数据(抽样后): %s, 总任务数: %s, 本节点GPU数: %s, 每GPU任务数: %s, 每任务数据量: %s", # 修改日志
        total_data,
        effective_total_tasks,
        gpus_on_this_node,
        args.tasks_per_gpu,
        data_per_task,
    )

    keys_in_order = list(data.keys()) if isinstance(data, dict) else None
    tasks_created_count = 0
    # 将列表或字典键转为列表，以便按索引分割
    data_items = list(data.items()) if isinstance(data, dict) else data

    for gpu_id in range(0, gpus_on_this_node, group_size):
        for task_id in range(args.tasks_per_gpu):
            group_idx = gpu_id // group_size
            global_task_idx = args.node_rank * effective_tasks_per_node + group_idx * args.tasks_per_gpu + task_id
            start_index = global_task_idx * data_per_task
            end_index = min(start_index + data_per_task, total_data)

            if start_index >= total_data:
                current_chunk_data_items = []
            else:
                current_chunk_data_items = data_items[start_index:end_index]

            # 将分割后的数据转回原始格式 (list 或 dict)
            if isinstance(data, dict):
                current_chunk_data = dict(current_chunk_data_items)
            else:
                current_chunk_data = current_chunk_data_items

            data_name = join(parallel_data_dir, f"{args.node_rank}_{gpu_id}_{task_id}.json")
            save_json(data_name, current_chunk_data)
            tasks_created_count += 1

    logger.info(f"分割数据集 {question_file} 完成。创建了 {tasks_created_count} 个任务文件。")
    return True


def parallel_config(args, task_config, model_config):
    """为每个任务创建独立的配置文件"""
    inference_dir = model_config["output_dir"]
    inference_dir = join(inference_dir, "inference", args.task)
    remote_answers_dir = join(model_config.get('remote_work_dir', model_config["output_dir"]), "inference", args.task)

    model_name_or_path = model_config.get("model_name_or_path") or task_config.get("model_name_or_path")
    if not model_name_or_path:
        logger.error("model_name_or_path 未在主配置或任务配置中指定。")
        return False

    makedirs(inference_dir, exist_ok=True)

    # 基础配置：合并 task_config 和 model_config，命令行参数稍后覆盖
    base_config = copy.deepcopy(task_config)
    # 将 model_config 中的关键设置（如模型路径、量化等）合并到 base_config
    keys_from_model_config = [
        "model_name_or_path", "quantization_bit", "device_map", "trust_remote_code",
        "max_memory",
        "prm_model", "prm_device_map", "prm_model_type", "infer_dtype",
        "image_resolution", "template", "print_param_status",
        # 从主配置继承生成参数，但会被命令行覆盖
        "temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty",
        "candidate_num", "mcts_iterations", "mcts_exploration_factor",
        "max_sampling_count", "atomthink_beam_search_num", "do_sample"
    ]
    for key in keys_from_model_config:
        if key in model_config:
            base_config[key] = model_config[key]

    # 确保 generating_args 存在
    if "generating_args" not in base_config or not isinstance(base_config.get("generating_args"), dict):
        base_config["generating_args"] = {}

    # 将生成相关的参数从顶层移入 generating_args (如果它们存在于顶层)
    for key in [
        "temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty",
        "candidate_num", "mcts_iterations", "mcts_exploration_factor",
        "max_sampling_count", "atomthink_beam_search_num", "method", "do_sample"
    ]:
        if key in base_config:
            # 检查目标字典中是否已存在该键，避免覆盖更具体的设置
            if key not in base_config["generating_args"]:
                base_config["generating_args"][key] = base_config.pop(key)
            else:
                # 如果 generating_args 中已有，则移除顶层的，以 generating_args 中的为准
                base_config.pop(key)

    base_config['model_name_or_path'] = model_name_or_path # 确保模型路径正确设置

    curPath = os.getcwd()
    parallel_config_dir = join(curPath, "configs", "parallel", args.task)
    makedirs(parallel_config_dir, exist_ok=True)

    gpus_on_this_node = args.nproc_per_node
    configs_created_count = 0
    group_size = max(args.gpus_per_task, 1)

    for gpu_id in range(0, gpus_on_this_node, group_size):
        for task_id in range(args.tasks_per_gpu):
            save_config = copy.deepcopy(base_config) # 每个任务基于合并后的基础配置

            dataset_filename = f"{args.node_rank}_{gpu_id}_{task_id}.json"
            dataset_path_relative = join(args.task, dataset_filename) # 相对于 parallel 目录

            full_dataset_path = join(curPath, "data", "test_data", "parallel", dataset_path_relative)
            if not isfile(full_dataset_path):
                logger.warning(f"分割后的数据集文件未找到 {full_dataset_path} (任务 {args.node_rank}_{gpu_id}_{task_id})。跳过此任务的配置生成。")
                continue

            # 检查分割后的数据集是否为空
            try:
                split_data = read_json(full_dataset_path)
                if not split_data: # 检查是否为空列表或空字典
                    logger.warning(f"分割后的数据集文件 {full_dataset_path} 为空。跳过此任务的配置生成。")
                    continue
            except Exception as e:
                logger.error(f"读取分割文件 {full_dataset_path} 失败: {e}。跳过。")
                continue


            # ---【修正 answers_file 路径】---
            task_inference_dir = join(inference_dir, f"{args.node_rank}_{gpu_id}_{task_id}")
            task_remote_dir = join(remote_answers_dir, f"{args.node_rank}_{gpu_id}_{task_id}")
            makedirs(task_inference_dir, exist_ok=True)
            makedirs(task_remote_dir, exist_ok=True)

            save_config["dataset_dir"] = join("data", "test_data", "parallel")
            save_config["dataset"] = dataset_path_relative
            save_config["answers_file"] = task_inference_dir
            save_config["remote_answers_file"] = task_remote_dir
            # ---【结束修正】---

            save_config["max_samples"] = None

            # ---【第二次修正 method 参数传递】---
            # 确保 generating_args 字典存在
            if "generating_args" not in save_config or not isinstance(save_config.get("generating_args"), dict):
                save_config["generating_args"] = {}

            # 1. 将命令行参数放入 generating_args (覆盖YAML中的默认值)
            save_config["generating_args"]["method"] = args.method
            save_config["generating_args"]["temperature"] = args.temperature
            save_config["generating_args"]["candidate_num"] = args.candidate_num
            save_config["generating_args"]["mcts_iterations"] = args.mcts_iterations
            save_config["generating_args"]["mcts_exploration_factor"] = args.mcts_exploration_factor
            save_config["generating_args"]["max_sampling_count"] = args.max_sampling_count
            save_config["generating_args"]["atomthink_beam_search_num"] = args.atomthink_beam_search_num
            if "do_sample" in model_config:
                save_config["generating_args"].setdefault("do_sample", model_config["do_sample"])
            # 2. 【冗余添加】同时将 method 写入顶层 (以防 get_infer_args 优先读取顶层)
            save_config["method"] = args.method
            # ---【第二次修正结束】---

            # 这两个参数可能属于顶层或 data_args
            save_config["prompt"] = args.prompt
            save_config["separate_eval"] = args.separate_eval

            # --- GPU 映射 ---
            base_device_map = model_config.get("device_map", "auto")
            if args.gpus_per_task > 1:
                # When one task spans multiple visible GPUs, pin the policy model
                # to the first device (cuda:0 within the local visibility scope)
                save_config["device_map"] = "cuda:0"
            else:
                if isinstance(base_device_map, dict):
                    save_config["device_map"] = copy.deepcopy(base_device_map)
                else:
                    save_config["device_map"] = base_device_map

            if "prm_device_map" in model_config:
                save_config["prm_device_map"] = model_config["prm_device_map"]
                logger.info(
                    "配置 %s_%s: 使用 device_map=%s, prm_device_map=%s (来自基础配置)",
                    gpu_id,
                    task_id,
                    save_config["device_map"],
                    save_config["prm_device_map"],
                )
            elif args.gpus_per_task > 1:
                reward_local_gpu = max(0, args.gpus_per_task - 1)
                save_config["prm_device_map"] = f"cuda:{reward_local_gpu}"
                logger.info(
                    "配置 %s_%s: 使用 device_map=%s, prm_device_map=%s (组大小=%s)",
                    gpu_id,
                    task_id,
                    save_config["device_map"],
                    save_config["prm_device_map"],
                    args.gpus_per_task,
                )
            else:
                save_config["prm_device_map"] = "auto"
                logger.info(
                    "配置 %s_%s: 使用 device_map=%s, prm_device_map=%s",
                    gpu_id, task_id,
                    save_config["device_map"], save_config["prm_device_map"]
                )

            # --- 保留显存与精度限制 ---
            if "max_memory" in model_config:
                save_config["max_memory"] = copy.deepcopy(model_config["max_memory"])
            if "infer_dtype" in model_config:
                save_config["infer_dtype"] = model_config["infer_dtype"]


            # 保存配置文件
            config_filename = f"{args.node_rank}_{gpu_id}_{task_id}.yaml"
            save_yaml(save_config, join(parallel_config_dir, config_filename))
            configs_created_count += 1

    logger.info(f"分割配置到 {parallel_config_dir} 完成。创建了 {configs_created_count} 个配置文件。")
    return configs_created_count > 0


def main():
    args = parse_args()
    print("解析得到的参数:", args)

    detected_gpu_count = detect_available_gpu_count()
    if detected_gpu_count is not None:
        if detected_gpu_count == 0:
            logger.warning(
                "检测到当前环境未暴露任何 GPU (CUDA_VISIBLE_DEVICES=%s)。继续运行可能失败。",
                os.environ.get("CUDA_VISIBLE_DEVICES")
            )
        elif args.nproc_per_node > detected_gpu_count:
            logger.warning(
                "nproc_per_node=%s 超过当前可见 GPU 数=%s，自动调整为 %s。",
                args.nproc_per_node,
                detected_gpu_count,
                detected_gpu_count,
            )
            args.nproc_per_node = detected_gpu_count
            if args.total_gpus and args.total_gpus > detected_gpu_count:
                logger.warning(
                    "total_gpus=%s 也超过可见 GPU 数，自动缩减为 %s。",
                    args.total_gpus,
                    detected_gpu_count,
                )
                args.total_gpus = detected_gpu_count

    # 加载主模型配置文件
    if not args.config or not isfile(args.config):
        logger.error(f"主配置文件 (--config) 未提供或未找到: {args.config}")
        return
    model_config = load_yaml(args.config)
    if model_config is None:
        logger.error(f"加载主配置文件失败: {args.config}")
        return

    # 使用命令行参数覆盖主配置
    if args.output_dir:
        model_config["output_dir"] = args.output_dir
        logger.warning("使用命令行的 output_dir 覆盖了配置文件中的设置。")
    if args.remote_work_dir:
        model_config["remote_work_dir"] = args.remote_work_dir
        logger.warning("使用命令行的 remote_work_dir 覆盖了配置文件中的设置。")

    # 验证必要的配置项
    if not model_config.get("output_dir"):
        logger.error("output_dir 未在配置或命令行中指定。")
        return
    # model_name_or_path 将在 parallel_config 中根据合并结果检查
    model_config.setdefault("remote_work_dir", model_config["output_dir"]) # 确保 remote_work_dir 有值

    log_dir = join(model_config["output_dir"], "logs")
    makedirs(log_dir, exist_ok=True)

    task_config = get_task_config(args.task)
    if task_config is None:
        logger.error(f"加载任务配置文件失败: task '{args.task}'")
        return

    # 准备数据集和配置文件
    if not parallel_dataset(args, task_config):
        logger.error("分割数据集失败。退出。")
        return
    if not parallel_config(args, task_config, model_config):
        logger.error("创建并行配置文件失败。退出。")
        return

    processes = []
    gpus_on_this_node = args.nproc_per_node

    # --- 启动子进程 ---
    group_size = max(args.gpus_per_task, 1)
    if group_size > gpus_on_this_node:
        logger.error(f"--gpus_per_task ({group_size}) 大于本节点 GPU 数 ({gpus_on_this_node})。")
        return
    if gpus_on_this_node % group_size != 0:
        logger.warning(
            "nproc_per_node=%s 不能被 gpus_per_task=%s 整除。最后一组 GPU 将被忽略。",
            gpus_on_this_node,
            group_size,
        )

    for gpu_id in range(0, gpus_on_this_node, group_size):
        gpu_indices = list(range(gpu_id, min(gpu_id + group_size, gpus_on_this_node)))
        if len(gpu_indices) < group_size:
            logger.warning(
                "GPU 组 %s 长度不足 %s，跳过启动任务。",
                gpu_indices,
                group_size,
            )
            continue
        for task_id in range(args.tasks_per_gpu):
            effective_gpu_id_for_files = gpu_id # 文件名和日志仍然使用基础 gpu_id

            # 设置 CUDA_VISIBLE_DEVICES
            visible_devices = ",".join(str(idx) for idx in gpu_indices)
            gpu_env = f"CUDA_VISIBLE_DEVICES={visible_devices}"

            # 构建路径
            log_file = join(log_dir, f"node{args.node_rank}_gpu_{effective_gpu_id_for_files}_{task_id}_task.log")
            config_file = join("configs", "parallel", args.task, f"{args.node_rank}_{effective_gpu_id_for_files}_{task_id}.yaml")

            # 检查配置文件是否存在
            if not isfile(config_file):
                logger.warning(f"配置文件 {config_file} 未找到。跳过启动任务。")
                continue

            # 构建命令
            command = f"{gpu_env} {CURRENT_PYTHON_PATH} src/llamafactory/evaluation/run_eval.py {config_file} 2>&1 | tee -a {log_file}"

            logger.info(f"启动任务: node_{args.node_rank}_gpu_{effective_gpu_id_for_files}_task_{task_id} (使用 GPU: {gpu_env.split('=')[1]})")
            logger.info(f"命令: {command}")
            try:
                # 启动进程
                process = subprocess.Popen(command, shell=True, executable="/bin/bash")
                processes.append(process)
            except Exception as e:
                logger.error(f"启动任务 {effective_gpu_id_for_files}_{task_id} 失败: {e}")

    # --- 等待所有进程结束 ---
    logger.info(f"等待所有 {len(processes)} 个已启动的任务完成...")
    failed_tasks = 0
    for i, p in enumerate(processes):
        try:
            return_code = p.wait() # 等待进程结束
            logger.info(f"任务 {i+1}/{len(processes)} 完成，返回码: {return_code}")
            if return_code != 0:
                logger.error(f"任务 {i+1} 失败! 返回码: {return_code}. 请检查对应的日志文件。") # 增加返回码信息
                failed_tasks += 1
                # 根据需要决定是否在第一个失败时就停止
                # break
        except Exception as e:
            logger.error(f"等待进程 {i+1} 时出错: {e}")
            failed_tasks += 1

    if failed_tasks > 0:
        logger.error(f"{failed_tasks} 个任务失败。")
    else:
        logger.info(f"所有任务成功完成!")

    logger.info(f"任务 '{args.task}' 结束。")

if __name__ == '__main__':
    main()
