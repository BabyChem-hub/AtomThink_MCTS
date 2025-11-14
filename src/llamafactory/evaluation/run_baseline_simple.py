import sys
from pathlib import Path
from os.path import dirname, abspath
import yaml
import fire

# Ensure project root 在 sys.path
current_dir = dirname(abspath(__file__))
sys.path.append(dirname(dirname(current_dir)))
sys.path.append(dirname(dirname(dirname(current_dir))))

from llamafactory.hparams import get_infer_args
from llamafactory.extras.logging import get_logger
from llamafactory.evaluation.method.base import BaseInference


logger = get_logger(__name__)


def run(
    config_file: str,
    override_model_path: str = None,
    max_samples: int = None,
    temperature: float = None,
    output_dir: str = None,
):
    """
    运行不依赖 PRM/MCTS 的基础推理。

    Args:
        config_file: 评估配置文件路径。
        override_model_path: 可选，覆盖配置里的模型权重目录。
    """
    args = yaml.safe_load(Path(config_file).read_text())
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)

    if override_model_path:
        model_args.model_name_or_path = override_model_path
        logger.info(f"Override model path: {override_model_path}")

    if max_samples is not None:
        data_args.max_samples = int(max_samples)
        logger.info(f"Override max_samples: {data_args.max_samples}")

    if temperature is not None:
        generating_args.temperature = float(temperature)
        logger.info(f"Override temperature: {generating_args.temperature}")

    if output_dir is not None:
        data_args.output_dir = output_dir
        logger.info(f"Override output_dir: {data_args.output_dir}")

    if not getattr(data_args, "answers_file", None):
        if not getattr(data_args, "output_dir", None):
            raise ValueError("data_args.answers_file 和 output_dir 至少需要提供一个。")
        data_args.answers_file = data_args.output_dir

    generating_args.method = "base"
    logger.info("Running baseline inference with base method (no PRM / no MCTS).")

    runner = BaseInference(model_args, data_args, generating_args, finetuning_args)
    runner.run_inference()
    runner.save_and_print_configs()
    logger.info("Baseline inference finished.")


if __name__ == "__main__":
    fire.Fire(run)
