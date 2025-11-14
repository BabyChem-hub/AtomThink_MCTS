import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _load_results(result_path: Path) -> List[Dict[str, Any]]:
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized result format in {result_path}")


def _split_rewards(entries: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    correct_rewards: List[float] = []
    incorrect_rewards: List[float] = []
    for item in entries:
        top_trajs = item.get("mcts_top_trajectories") or []
        if not top_trajs:
            continue
        reward = top_trajs[0].get("avg_value")
        if reward is None:
            continue
        if item.get("score") == 1:
            correct_rewards.append(float(reward))
        else:
            incorrect_rewards.append(float(reward))
    return correct_rewards, incorrect_rewards


def summarize_results(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(entries)
    correct = sum(1 for item in entries if item.get("score") == 1)
    iter_limit = sum(
        1 for item in entries if item.get("total_inference_times") == item.get("max_mcts_iterations")
    )
    fail_try = sum(1 for item in entries if item.get("resolved_via_fail_try"))
    fail_correct = sum(
        1
        for item in entries
        if item.get("resolved_via_fail_try") and item.get("score") == 1
    )
    early_stop = sum(1 for item in entries if item.get("early_stop_reason"))

    correct_rewards, incorrect_rewards = _split_rewards(entries)
    stats = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": (correct / total) if total else 0.0,
        "iter_limit": iter_limit,
        "fail_try": fail_try,
        "fail_try_correct": fail_correct,
        "fail_try_incorrect": fail_try - fail_correct,
        "early_stop": early_stop,
        "reward_correct_avg": mean(correct_rewards) if correct_rewards else None,
        "reward_incorrect_avg": mean(incorrect_rewards) if incorrect_rewards else None,
    }
    return stats


def _resolve_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_overview_image(stats: Dict[str, Any], output_path: Path) -> None:
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    header_font = _resolve_font(34)
    body_font = _resolve_font(26)

    lines = [
        "MCTS 运行统计概览",
        "",
        f"总题目数: {stats['total']}",
        f"正确题目: {stats['correct']}  |  错误题目: {stats['incorrect']}",
        f"准确率: {stats['accuracy']:.2%}",
        "",
        f"达到迭代上限: {stats['iter_limit']}",
        f"触发 fail_try(): {stats['fail_try']}  (修正成功 {stats['fail_try_correct']}, 仍然错误 {stats['fail_try_incorrect']})",
        f"提前停止 (early stop): {stats['early_stop']}",
        "",
        "平均 Reward（Top 轨迹）:",
    ]

    reward_correct = stats.get("reward_correct_avg")
    reward_incorrect = stats.get("reward_incorrect_avg")
    if reward_correct is not None or reward_incorrect is not None:
        lines.append(
            f"  正确: {reward_correct:.4f}" if reward_correct is not None else "  正确: N/A"
        )
        lines.append(
            f"  错误: {reward_incorrect:.4f}" if reward_incorrect is not None else "  错误: N/A"
        )
    else:
        lines.append("  Reward 数据不可用")

    lines.extend(
        [
            "",
            "流程回顾: Selection → Expansion → Simulation/Backprop →",
            "终结节点筛选 → 强制终答/包装 → fail_try() 兜底",
        ]
    )

    y = 50
    for idx, line in enumerate(lines):
        font = header_font if idx == 0 else body_font
        draw.text((50, y), line, fill="black", font=font)
        y += 44 if idx == 0 else 36

    image.save(output_path)


def _load_trace_file(trace_path: Path) -> Dict[str, Any]:
    with trace_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_render_tree():
    try:
        if __package__:
            from .mcts_tree_viz import render_tree  # type: ignore
        else:
            src_dir = Path(__file__).resolve().parents[3]
            if str(src_dir) not in sys.path:
                sys.path.append(str(src_dir))
            from llamafactory.evaluation.utils.mcts_tree_viz import render_tree  # type: ignore
        return render_tree
    except ImportError as exc:
        raise RuntimeError("无法导入树状可视化模块 mcts_tree_viz，请确认文件存在。") from exc


def _render_trace_mode(
    trace_path: Path,
    output: Optional[Path],
    max_nodes: int,
    exploration: float,
    h_spacing: int,
    v_spacing: int,
) -> Path:
    render_tree = _resolve_render_tree()
    trace = _load_trace_file(trace_path)
    output_path = output or trace_path.with_name(trace_path.stem + "_tree.png")
    render_tree(
        trace,
        output_path,
        max_nodes=max_nodes,
        exploration_factor=exploration,
        h_spacing=h_spacing,
        v_spacing=v_spacing,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 MCTS 推理可视化（概览或单样本搜索树）")
    parser.add_argument(
        "--mode",
        choices=["overview", "trace"],
        default="overview",
        help="overview 输出整体统计，trace 输出单个样本的搜索树",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="overview 模式：推理输出 JSON（answers_mcts_xxx.json）路径",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="trace 模式：单个样本的 mcts_trace_xxx.json 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 PNG 文件路径；缺省时自动生成",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=120,
        help="trace 模式：图中最多展示的节点数量",
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=1.41421356237,
        help="trace 模式：UCT 计算使用的探索系数 c",
    )
    parser.add_argument(
        "--h-spacing",
        type=int,
        default=220,
        help="trace 模式：节点水平间距",
    )
    parser.add_argument(
        "--v-spacing",
        type=int,
        default=150,
        help="trace 模式：节点垂直间距",
    )
    args = parser.parse_args()

    if args.mode == "overview":
        if args.results is None:
            parser.error("--results 在 overview 模式必须提供")
        entries = _load_results(args.results)
        stats = summarize_results(entries)
        output_path = (
            args.output
            if args.output is not None
            else args.results.with_name(args.results.stem + "_overview.png")
        )
        render_overview_image(stats, output_path)
        print(f"统计图已生成: {output_path}")
        return

    if args.trace is None:
        parser.error("--trace 在 trace 模式必须提供")
    output_path = _render_trace_mode(
        trace_path=args.trace,
        output=args.output,
        max_nodes=args.max_nodes,
        exploration=args.exploration,
        h_spacing=args.h_spacing,
        v_spacing=args.v_spacing,
    )
    print(f"搜索树已生成: {output_path}")


if __name__ == "__main__":
    main()
