import os
import fire
import yaml
from pathlib import Path
import sys
from os.path import dirname

# Make sure project root is in python path
current_dir = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(current_dir)))
sys.path.append(dirname(dirname(dirname(current_dir))))

from llamafactory.hparams import get_infer_args
from llamafactory.extras.logging import get_logger
from llamafactory.evaluation.method.base import BaseInference # <-- This line was uncommented
from llamafactory.evaluation.method.atomthink_quick import AtomThinkQuick
from llamafactory.evaluation.method.atomthink_slow import AtomThinkSlow
from llamafactory.evaluation.method.atomthink_mcts import AtomThinkMCTS
from llamafactory.evaluation.method.atomic_scoring import AtomicScoring

logger = get_logger(__name__)


def evaluation(config_file: str):
    """
    Main function to run evaluation based on a YAML config file.

    Args:
        config_file (str): Path to the YAML configuration file.
    """
    args = yaml.safe_load(Path(config_file).read_text())
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)

    logger.info(f"Answers will be saved to: {data_args.answers_file}")
    if hasattr(data_args, "remote_answers_file"):
         logger.info(f"Remote answers file: {data_args.remote_answers_file}")

    method = generating_args.method
    logger.info(f"Using evaluation method: {method}")

    if method in ["base", "cot"]:
        eval_func = BaseInference(model_args, data_args, generating_args, finetuning_args)
    elif method == "quick":
        eval_func = AtomThinkQuick(model_args, data_args, generating_args, finetuning_args)
    elif method == "slow":
        eval_func = AtomThinkSlow(model_args, data_args, generating_args, finetuning_args)
    elif method == "mcts":
        eval_func = AtomThinkMCTS(model_args, data_args, generating_args, finetuning_args)
    elif method == "atomic":
        eval_func = AtomicScoring(model_args, data_args, generating_args, finetuning_args)
    else:
        raise ValueError(f"Unsupported inference method: {method}")

    eval_func.run_inference()
    logger.info("Evaluation finished.")


if __name__ == '__main__':
    fire.Fire(evaluation)
