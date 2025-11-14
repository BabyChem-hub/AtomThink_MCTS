import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_EXPLORATION = 1.41421356237
EPSILON = 1e-6


def _load_trace(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _collect_first_visit(events: List[Dict]) -> Dict[int, int]:
    first_visit: Dict[int, int] = {}
    for event in events:
        iteration = event.get("iteration")
        for node_id in event.get("expanded_nodes", []):
            if node_id is not None and node_id not in first_visit:
                first_visit[node_id] = iteration
        simulated = event.get("simulated_node")
        if simulated is not None and simulated not in first_visit:
            first_visit[simulated] = iteration
    return first_visit


def _collect_reward_stats(events: List[Dict]) -> Dict[int, List[float]]:
    reward_map: Dict[int, List[float]] = {}
    for event in events:
        node_id = event.get("simulated_node")
        if node_id is None:
            continue
        reward = event.get("reward")
        if reward is None:
            continue
        reward_map.setdefault(node_id, []).append(float(reward))
    return reward_map


def _select_nodes(nodes: List[Dict], best_path_ids: List[int], max_nodes: int) -> Set[int]:
    selected: Set[int] = set(best_path_ids)
    id_to_parent = {entry["id"]: entry.get("parent_id") for entry in nodes}
    if nodes:
        selected.add(nodes[0]["id"])  # Fallback root
    sorted_nodes = sorted(nodes, key=lambda item: item.get("visits", 0), reverse=True)
    for entry in sorted_nodes:
        node_id = entry["id"]
        if node_id in selected:
            continue
        selected.add(node_id)
        parent_id = id_to_parent.get(node_id)
        while parent_id is not None and parent_id not in selected:
            selected.add(parent_id)
            parent_id = id_to_parent.get(parent_id)
        if len(selected) >= max_nodes:
            break
    return selected


def _build_children_map(nodes: List[Dict]) -> Tuple[Dict[int, List[int]], Optional[int]]:
    children: Dict[int, List[int]] = {}
    root_id: Optional[int] = None
    for entry in nodes:
        node_id = entry["id"]
        parent_id = entry.get("parent_id")
        if parent_id is None:
            root_id = node_id
        else:
            children.setdefault(parent_id, []).append(node_id)
        children.setdefault(node_id, [])
    return children, root_id


def _assign_positions(node_id: int, children_map: Dict[int, List[int]], depth_map: Dict[int, int],
                      selected: Set[int], next_leaf: List[float], positions: Dict[int, float]) -> float:
    selected_children = [child for child in children_map.get(node_id, []) if child in selected]
    if not selected_children:
        pos = next_leaf[0]
        next_leaf[0] += 1.0
    else:
        child_positions = [
            _assign_positions(child, children_map, depth_map, selected, next_leaf, positions)
            for child in selected_children
        ]
        pos = sum(child_positions) / len(child_positions)
    positions[node_id] = pos
    return pos


def _compute_depths(root_id: int, children_map: Dict[int, List[int]]) -> Dict[int, int]:
    depth_map: Dict[int, int] = {root_id: 0}
    queue = [root_id]
    while queue:
        current = queue.pop(0)
        for child in children_map.get(current, []):
            depth_map[child] = depth_map[current] + 1
            queue.append(child)
    return depth_map


def _build_node_histories(events: List[Dict]) -> Dict[int, List[Dict]]:
    histories: Dict[int, List[Dict]] = {}
    for event in events:
        iteration = event.get("iteration")
        reward = event.get("reward")

        for node_id in event.get("expanded_nodes", []):
            if node_id is None:
                continue
            histories.setdefault(node_id, []).append(
                {
                    "type": "expansion",
                    "iteration": iteration,
                    "detail": "节点在该轮被扩展",
                }
            )

        for step in event.get("selection_steps", []) or []:
            chosen_id = step.get("selected_child_id")
            if chosen_id is None:
                continue
            chosen_child = None
            for child in step.get("children", []):
                if child.get("node_id") == chosen_id:
                    chosen_child = child
                    break
            histories.setdefault(chosen_id, []).append(
                {
                    "type": "selection",
                    "iteration": iteration,
                    "reason": step.get("reason"),
                    "uct": chosen_child.get("uct") if chosen_child else None,
                    "rank": chosen_child.get("rank") if chosen_child else None,
                    "detail": "选择阶段命中该节点",
                }
            )

        for entry in event.get("backprop", []) or []:
            node_id = entry.get("node_id")
            if node_id is None:
                continue
            histories.setdefault(node_id, []).append(
                {
                    "type": "backprop",
                    "iteration": iteration,
                    "reward": reward,
                    "visits_before": entry.get("visits_before"),
                    "visits_after": entry.get("visits_after"),
                    "value_before": entry.get("value_before"),
                    "value_after": entry.get("value_after"),
                    "detail": "回传更新",
                }
            )

    for entries in histories.values():
        entries.sort(key=lambda item: item.get("iteration", -1))
    return histories


def _format_path(path: List[int]) -> str:
    if not path:
        return "-"
    return "→".join(str(pid) for pid in path)


def _build_iteration_summary(events: List[Dict], limit: int = 18) -> List[str]:
    rows: List[str] = []
    for event in events:
        iteration = event.get("iteration")
        path = event.get("selected_path") or []
        simulated = event.get("simulated_node")
        reward = event.get("reward")
        node_visits = event.get("node_visits")
        node_value = event.get("node_value")

        reason = None
        uct = None
        for step in event.get("selection_steps", []) or []:
            if step.get("selected_child_id") == simulated:
                reason = step.get("reason")
                for child in step.get("children", []):
                    if child.get("node_id") == simulated:
                        uct = child.get("uct")
                        break
                break

        summary = (
            f"迭代 {iteration}: 路径 {_format_path(path)} → 节点 {simulated} "
            f"(理由: {reason or 'N/A'}, UCT: {_format_float(uct)}) | "
            f"reward={_format_float(reward)} | visits={node_visits} | value={_format_float(node_value)}"
        )
        rows.append(summary)

    if len(rows) > limit:
        remaining = len(rows) - limit
        rows = rows[:limit]
        rows.append(f"... 还有 {remaining} 条迭代记录未显示")
    return rows


def _wrap_text(text: str, width: int) -> List[str]:
    if not text:
        return []
    wrapped = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)
    return wrapped or [text]


def _format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
        return f"{value:.{precision}f}"
    return str(value)


def _compute_uct(
    node: Dict,
    parent: Optional[Dict],
    exploration_factor: float,
) -> Optional[float]:
    visits = node.get("visits", 0) or 0
    value = float(node.get("value", 0.0) or 0.0)
    if visits == 0:
        return math.inf
    parent_visits = (parent or {}).get("visits", 0) or 0
    parent_visits = max(parent_visits, 1)
    return (value / (visits + EPSILON)) + exploration_factor * math.sqrt(
        math.log(parent_visits + EPSILON) / (visits + EPSILON)
    )


def _build_best_path_pairs(best_path_ids: Sequence[int]) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    for idx in range(len(best_path_ids) - 1):
        pairs.add((best_path_ids[idx], best_path_ids[idx + 1]))
    return pairs


def render_tree(
    trace: Dict,
    output_path: Path,
    max_nodes: int = 80,
    exploration_factor: float = DEFAULT_EXPLORATION,
    h_spacing: int = 220,
    v_spacing: int = 150,
) -> None:
    nodes = trace.get("nodes", [])
    events = trace.get("events", [])
    best_path_ids = trace.get("best_path_ids", [])
    children_map, root_id = _build_children_map(nodes)
    if root_id is None and nodes:
        root_id = nodes[0]["id"]
    if root_id is None:
        raise ValueError("Trace file does not contain root information.")

    depth_map = _compute_depths(root_id, children_map)
    selected = _select_nodes(nodes, best_path_ids, max_nodes)
    selected.add(root_id)

    reward_map = _collect_reward_stats(events)
    history_map = _build_node_histories(events)
    timeline_rows = _build_iteration_summary(events)
    node_lookup = {entry["id"]: entry for entry in nodes}
    first_visit = _collect_first_visit(events)
    positions: Dict[int, float] = {}
    _assign_positions(root_id, children_map, depth_map, selected, [0.0], positions)

    min_pos = min(positions.values())
    max_pos = max(positions.values())
    max_depth = max(depth_map.get(node_id, 0) for node_id in selected)

    small_font = _resolve_font(20)
    sample_bbox = small_font.getbbox("Ag")
    dynamic_line_height = (sample_bbox[3] - sample_bbox[1]) + 6
    title_font = _resolve_font(34)
    body_font = _resolve_font(22)
    timeline_font = _resolve_font(20)

    margin_x = 130
    margin_top = 140
    box_width = 280
    base_box_min_height = 170
    line_height = max(dynamic_line_height, 28)

    node_render_info: Dict[int, Dict[str, any]] = {}
    for node_id in selected:
        node = node_lookup.get(node_id, {})
        visits = int(node.get("visits", 0) or 0)
        value = float(node.get("value", 0.0) or 0.0)
        step_text = node.get("step") or ""
        is_terminal = node.get("is_terminal", False)
        first_iter = first_visit.get(node_id, "-")
        parent = node_lookup.get(node.get("parent_id"))
        avg_value = value / visits if visits else None
        uct = _compute_uct(node, parent, exploration_factor) if parent is not None else None
        rewards = reward_map.get(node_id)
        last_reward = rewards[-1] if rewards else None
        mean_reward = sum(rewards) / len(rewards) if rewards else None
        history_entries = history_map.get(node_id, [])
        history_lines: List[str] = []
        history_limit = 3
        for entry in history_entries[:history_limit]:
            iter_id = entry.get("iteration")
            if entry.get("type") == "expansion":
                history_lines.append(f"Iter{iter_id}: 扩展产生")
            elif entry.get("type") == "selection":
                history_lines.append(
                    f"Iter{iter_id}: 选择(理由={entry.get('reason')}, UCT={_format_float(entry.get('uct'))})"
                )
            elif entry.get("type") == "backprop":
                visits_before = entry.get("visits_before")
                visits_after = entry.get("visits_after")
                history_lines.append(
                    f"Iter{iter_id}: 回传 {visits_before}->{visits_after} (reward={_format_float(entry.get('reward'))})"
                )
        if len(history_entries) > history_limit:
            history_lines.append(f"... (+{len(history_entries) - history_limit})")

        text_lines = [
            f"ID:{node_id} Depth:{depth_map.get(node_id, 0)} Terminal:{is_terminal}",
            f"Visits:{visits} AvgV:{_format_float(avg_value)} Value:{_format_float(value)}",
        ]
        if parent is not None:
            text_lines.append(
                f"UCT:{'∞' if uct is not None and not math.isfinite(uct) else _format_float(uct)} FirstIter:{first_iter}"
            )
        else:
            text_lines.append(f"FirstIter:{first_iter}")
        text_lines.append(f"Reward(last/mean): {_format_float(last_reward)} / {_format_float(mean_reward)}")
        if history_lines:
            text_lines.append("History:")
            text_lines.extend(history_lines)
        if step_text:
            snippet = step_text.strip().replace("\n", " ")
            wrapped = _wrap_text(snippet, 36)
            for idx, part in enumerate(wrapped[:3]):
                prefix = "Step: " if idx == 0 else "      "
                text_lines.append(f"{prefix}{part}")
            if len(wrapped) > 3:
                text_lines.append("      ...")

        content_height = 20 + len(text_lines) * line_height
        node_render_info[node_id] = {
            "text_lines": text_lines,
            "box_height": max(base_box_min_height, content_height),
            "is_terminal": is_terminal,
        }

    depth_height_map: Dict[int, int] = {}
    for node_id in selected:
        depth = depth_map.get(node_id, 0)
        box_height = node_render_info.get(node_id, {}).get("box_height", base_box_min_height)
        depth_height_map[depth] = max(depth_height_map.get(depth, base_box_min_height), box_height)
    if not depth_height_map:
        depth_height_map[0] = base_box_min_height
    sorted_depths = sorted(depth_height_map.keys())
    depth_positions: Dict[int, float] = {}
    min_gap_between_layers = 40
    for idx, depth in enumerate(sorted_depths):
        baseline_top = margin_top + depth * v_spacing
        if idx == 0:
            depth_positions[depth] = baseline_top
            continue
        prev_depth = sorted_depths[idx - 1]
        prev_bottom = depth_positions[prev_depth] + depth_height_map[prev_depth]
        desired_top = prev_bottom + min_gap_between_layers
        depth_positions[depth] = max(baseline_top, desired_top)

    width = int((max_pos - min_pos + 1) * h_spacing + margin_x * 2)
    timeline_lines: List[str] = []
    timeline_block_height = 0
    if timeline_rows:
        for row in timeline_rows:
            timeline_lines.extend(_wrap_text(row, 64))
        timeline_block_height = 80 + len(timeline_lines) * 26

    tree_bottom = max(
        (depth_positions[depth] + depth_height_map[depth] for depth in sorted_depths),
        default=margin_top + base_box_min_height,
    )
    content_bottom = tree_bottom + (timeline_block_height if timeline_lines else 0) + (60 if timeline_lines else 40)
    image_height = int(max(content_bottom, 600))

    image = Image.new("RGB", (max(width, 900), image_height), "white")
    draw = ImageDraw.Draw(image)

    header_lines = [
        f"Sample {trace.get('sample_index', 'N/A')} | Pos {trace.get('dataset_position', 'N/A')}",
        f"Answer: {trace.get('answer', 'N/A')} | Model: {trace.get('response', 'N/A')}",
        f"Early stop: {trace.get('early_stop_reason', 'None')} | fail_try: {trace.get('resolved_via_fail_try', False)}",
    ]
    header_lines.extend(
        [
            f"迭代次数: {len(events)} | 记录节点: {len(nodes)} | 展示节点: {len(selected)}",
            (
                f"最大深度: {max_depth} | 最佳路径长度: {len(best_path_ids)} | "
                f"UCT探索系数: {exploration_factor:.3f}"
            ),
            "颜色: 黄色=最佳路径, 粉红=终结节点, 灰色=普通节点, 橙色连线=最佳路径",
        ]
    )
    y_cursor = 30
    for idx, line in enumerate(header_lines):
        draw.text((40, y_cursor), line, fill="black", font=title_font if idx == 0 else body_font)
        y_cursor += 44 if idx == 0 else 32

    best_path_pairs = _build_best_path_pairs(best_path_ids)

    def project(node_id: int) -> Tuple[float, float]:
        pos = positions[node_id]
        depth = depth_map.get(node_id, 0)
        x = margin_x + (pos - min_pos) * h_spacing
        y = depth_positions.get(depth, margin_top + depth * v_spacing)
        return x, y

    # Draw edges first
    for parent_id, children_ids in children_map.items():
        if parent_id not in selected:
            continue
        parent_pos = project(parent_id)
        parent_height = node_render_info.get(parent_id, {}).get("box_height", base_box_min_height)
        for child_id in children_ids:
            if child_id not in selected:
                continue
            child_pos = project(child_id)
            edge = (parent_id, child_id)
            edge_color = "#f28f16" if edge in best_path_pairs else "#8c8c8c"
            edge_width = 5 if edge in best_path_pairs else 3
            draw.line(
                [
                    (parent_pos[0] + box_width / 2, parent_pos[1] + parent_height),
                    (child_pos[0] + box_width / 2, child_pos[1]),
                ],
                fill=edge_color,
                width=edge_width,
            )

    best_path_set = set(best_path_ids)
    for node_id in selected:
        node = node_lookup.get(node_id, {})
        x, y = project(node_id)
        render_info = node_render_info.get(node_id, {})
        box_height = render_info.get("box_height", base_box_min_height)
        is_terminal = render_info.get("is_terminal", False)
        text_lines = render_info.get("text_lines", [])

        box = [x, y, x + box_width, y + box_height]
        if is_terminal:
            fill = "#f8d7da"
        else:
            fill = "#f2f2f2"
        draw.rectangle(box, fill=fill, outline="#5f5f5f", width=3)

        if node_id in best_path_set:
            highlight_strip = [x - 16, y, x - 4, y + box_height]
            draw.rectangle(highlight_strip, fill="#f2a93b")
            draw.rectangle(box, outline="#f2a93b", width=4)

        text_y = y + 8
        for line in text_lines:
            draw.text((x + 12, text_y), line, fill="black", font=small_font)
            text_y += line_height

    if timeline_lines:
        timeline_y = tree_bottom + 30
        draw.text((40, timeline_y), "迭代过程回放", fill="black", font=title_font)
        timeline_y += 40
        for row in timeline_lines:
            draw.text((40, timeline_y), row, fill="black", font=timeline_font)
            timeline_y += 26

    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="将单个 MCTS trace JSON 转换为树状可视化图片")
    parser.add_argument("--trace", type=Path, required=True, help="trace JSON 文件路径")
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径")
    parser.add_argument("--max-nodes", type=int, default=80, help="图中最多展示的节点数量")
    parser.add_argument(
        "--exploration",
        type=float,
        default=DEFAULT_EXPLORATION,
        help="UCT 计算时使用的探索系数 c",
    )
    parser.add_argument("--h-spacing", type=int, default=220, help="节点水平间距")
    parser.add_argument("--v-spacing", type=int, default=150, help="节点垂直间距")
    args = parser.parse_args()

    trace = _load_trace(args.trace)
    output_path = args.output or args.trace.with_name(args.trace.stem + "_tree.png")
    render_tree(
        trace,
        output_path,
        max_nodes=args.max_nodes,
        exploration_factor=args.exploration,
        h_spacing=args.h_spacing,
        v_spacing=args.v_spacing,
    )
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
