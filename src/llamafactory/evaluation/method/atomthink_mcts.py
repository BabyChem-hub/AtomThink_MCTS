import time
import math
import random
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from ...extras.logging import get_logger
from ..utils.inference_utils import inference_one_step, update_inputs_from_rollout, txt_verifier, has_final_answer_marker
from ..utils.eval_utils import save_json
from ..utils.conversation import conversation_map
from .base import BaseInference
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F
import copy

logger = get_logger(__name__)

# 为 MCTS 添加的节点类
class MCTSNode:
    """
    蒙特卡洛树搜索的节点类
    """
    def __init__(self, state, parent=None):
        self.state = state  # 当前节点的推理步骤列表 (rollout)
        self.parent = parent
        self.children = []
        self.visits = 0  # 访问次数
        self.value = 0.0  # 累积奖励
        self.is_terminal = False # 是否为终结节点
        self.untried_actions = None # 尚未尝试的动作
        self.node_id = None

    def uct(self, exploration_factor=1.414):
        """
        计算 UCT (Upper Confidence Bound for Trees) 值，用于选择节点
        """
        if self.visits == 0:
            return float('inf') # 优先探索未访问过的节点
        # UCT = exploitation_term + exploration_term
        # Add a small epsilon to parent visits to avoid division by zero or log(0) if parent has 0 visits (though root shouldn't)
        parent_visits = self.parent.visits if self.parent else 1
        epsilon = 1e-6
        return (self.value / (self.visits + epsilon)) + exploration_factor * math.sqrt(
            math.log(parent_visits + epsilon) / (self.visits + epsilon)
        )

class AtomThinkMCTS(BaseInference):
    def __init__(self, model_args, data_args, generating_args, finetuning_args):
        super().__init__(model_args, data_args, generating_args, finetuning_args)
        self.prm_model, self.prm_tokenizer, self.candidate_tokens, self.step_tag_id = self.load_reward_model(
            self.model_args)
        self.prm_device = self._resolve_prm_device(self.prm_model)
        trace_samples_raw = getattr(self.generating_args, "mcts_trace_samples", None)
        self.mcts_trace_samples = set()
        if trace_samples_raw:
            if isinstance(trace_samples_raw, str):
                tokens = [token.strip() for token in re.split(r"[,\s]+", trace_samples_raw) if token.strip()]
                self.mcts_trace_samples = set(tokens)
            elif isinstance(trace_samples_raw, (list, tuple, set)):
                self.mcts_trace_samples = {str(token) for token in trace_samples_raw if str(token).strip()}
            else:
                logger.warning(f"Unsupported mcts_trace_samples type: {type(trace_samples_raw)}")
        trace_dir_raw = getattr(self.generating_args, "mcts_trace_dir", None)
        if trace_dir_raw:
            self.mcts_trace_dir = Path(trace_dir_raw)
        else:
            answers_file = getattr(self.data_args, "answers_file", None)
            if answers_file:
                self.mcts_trace_dir = Path(answers_file).parent / "mcts_traces"
            else:
                self.mcts_trace_dir = Path("mcts_traces")
        self.mcts_trace_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_final_answer_prompt(self, input_item):
        prompt = input_item.get("prompt", "")
        guidance = (
            "\n\nWhen you finish reasoning, you must conclude with "
            "\"To sum up, the final answer is <answer>.\""
        )
        if prompt and "final answer is" not in prompt:
            updated_prompt = prompt.rstrip() + guidance
            images = input_item.get("images")
            input_item["prompt"] = updated_prompt
            input_item["inputs"] = self.processor(
                text=[updated_prompt],
                images=images,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
        return input_item

    def _force_final_answer_generation(self, input_item, rollout):
        if not rollout:
            return None
        try:
            forced_item = update_inputs_from_rollout(copy.deepcopy(input_item), rollout, self.processor)
            instruction = (
                "\n\nPlease conclude now with "
                "\"To sum up, the final answer is <answer>.\""
            )
            question_type = self._get_question_type(forced_item.get("metadata"))
            if question_type in {"multi-choice", "multi_choice", "multiple-choice"}:
                instruction += (
                    " Respond with only the correct option letter (A, B, C, or D) when giving the final answer."
                )
            forced_prompt = forced_item["prompt"].rstrip() + instruction
            forced_item["prompt"] = forced_prompt
            forced_item["inputs"] = self.processor(
                text=[forced_prompt],
                images=forced_item.get("images"),
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            response = inference_one_step(
                forced_item["inputs"],
                self.model,
                self.processor,
                self.generating_args,
                temperature=0.0,
            )
            response = self._normalize_multi_choice_response(response, forced_item.get("metadata"))
            return response
        except Exception as e:
            logger.error(f"Error during forced final answer generation: {e}")
            return None

    # --- 【新增函数】 开始 ---


    def _collect_top_trajectories(self, root_node, top_n=5):
        """
        遍历 MCTS 树，收集访问次数最多的 top_n 条完整轨迹 (rollouts)。
        返回每个轨迹及其访问次数和平均奖励。
        """
        if not root_node:
            return []

        # We collect nodes that represent potential final states or significant branches
        # A simple approach is to collect all nodes with visits > 0
        all_nodes_with_stats = []
        q = [root_node]
        visited = {root_node}

        while q:
            current = q.pop(0)
            # Only record nodes that were actually visited during the search
            if current.visits > 0:
                avg_value = current.value / current.visits if current.visits > 0 else 0
                all_nodes_with_stats.append({
                    "rollout": current.state, # The sequence of steps to reach this node
                    "visits": current.visits,
                    "avg_value": avg_value,
                    "is_terminal": current.is_terminal # Was this considered a terminal state?
                })

            for child in current.children:
                if child not in visited:
                    visited.add(child)
                    q.append(child)

        # 按访问次数降序排序
        all_nodes_with_stats.sort(key=lambda x: x["visits"], reverse=True)

        # 返回前 top_n 个
        # Filter out the root node itself if it's the only one visited and has children
        if len(all_nodes_with_stats) > 1 and all_nodes_with_stats[0]["rollout"] == []:
             return all_nodes_with_stats[1:top_n+1]
        else:
             return all_nodes_with_stats[:top_n]
    
    def _should_trace_sample(self, sample_index: Optional[str], position: int) -> bool:
        if not self.mcts_trace_samples:
            return False
        lowered_tokens = {token.lower() for token in self.mcts_trace_samples}
        if "all" in lowered_tokens or "*" in lowered_tokens:
            return True
        candidates = {str(position)}
        if sample_index is not None:
            candidates.add(str(sample_index))
        return any(candidate in self.mcts_trace_samples for candidate in candidates)

    def _resolve_trace_path(self, sample_index: Optional[str], position: int) -> Path:
        identifier = str(sample_index) if sample_index is not None else f"idx_{position}"
        safe_identifier = re.sub(r"[^0-9A-Za-z_-]", "_", identifier)
        return self.mcts_trace_dir / f"mcts_trace_{safe_identifier}.json"

    @staticmethod
    def _trace_assign_node(trace_state: Dict[str, Any], node: Optional[MCTSNode]) -> Optional[int]:
        if node is None:
            return None
        node_id = trace_state["nodes"].get(node)
        if node_id is None:
            node_id = trace_state["next_id"]
            trace_state["nodes"][node] = node_id
            trace_state["next_id"] += 1
            node.node_id = node_id
        return node_id

    def _trace_collect_nodes(self, trace_state: Dict[str, Any], root: MCTSNode) -> List[Dict[str, Any]]:
        nodes_payload: List[Dict[str, Any]] = []
        queue: List[MCTSNode] = [root]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            node_id = self._trace_assign_node(trace_state, current)
            parent_id = self._trace_assign_node(trace_state, current.parent) if current.parent else None
            nodes_payload.append(
                {
                    "id": node_id,
                    "parent_id": parent_id,
                    "visits": current.visits,
                    "value": current.value,
                    "is_terminal": current.is_terminal,
                    "state_length": len(current.state),
                    "step": current.state[-1] if current.state else None,
                }
            )
            queue.extend(current.children)
        nodes_payload.sort(key=lambda item: item["id"] if item["id"] is not None else -1)
        return nodes_payload

    def _trace_path_ids(self, trace_state: Dict[str, Any], node: Optional[MCTSNode]) -> List[int]:
        if node is None:
            return []
        path_nodes: List[MCTSNode] = []
        current = node
        while current is not None:
            path_nodes.append(current)
            current = current.parent
        path_node_ids = [self._trace_assign_node(trace_state, n) for n in reversed(path_nodes)]
        return [nid for nid in path_node_ids if nid is not None]

    def _dump_trace_output(
        self,
        trace_state: Dict[str, Any],
        root: MCTSNode,
        output_path: Path,
        sample_payload: Dict[str, Any],
    ) -> None:
        payload = sample_payload.copy()
        payload["nodes"] = self._trace_collect_nodes(trace_state, root)
        payload["events"] = trace_state.get("events", [])
        save_json(output_path, payload)
    # --- 【新增函数】 结束 ---

    def _wrap_with_final_answer(self, rollout):
        if not rollout:
            return None
        for step in reversed(rollout):
            if not step:
                continue
            lines = [line.strip() for line in step.splitlines() if line.strip()]
            for line in reversed(lines):
                candidate = self._extract_answer_candidate(line)
                if candidate:
                    return f"To sum up, the final answer is: {candidate}"
        return None

    @staticmethod
    def _extract_answer_candidate(text):
        if not text:
            return None
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized or normalized.endswith("?"):
            return None
        normalized = re.sub(r"^(?:Step \d+:)\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"^(?:Therefore|Thus|Hence|So|Then|Finally|Answer|Result)\s*[:,]?\s*", "",
                            normalized, flags=re.IGNORECASE)
        boxed_match = re.search(r"\\boxed\{([^{}]+)\}", normalized)
        if boxed_match:
            normalized = boxed_match.group(1).strip()
        for pattern in (r":\s*(.+)$", r"\bis\s+(.+)$", r"=\s*(.+)$"):
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().strip(".")
                if candidate:
                    return candidate
        normalized = normalized.strip(".")
        return normalized if normalized else None

    @staticmethod
    def _extract_choice_from_response(response: str) -> Optional[str]:
        if not response:
            return None

        def _normalize_candidate(text: Optional[str]) -> Optional[str]:
            if not text:
                return None
            cleaned = re.sub(r"[^A-D]", "", text.upper())
            return cleaned if cleaned in {"A", "B", "C", "D"} else None

        # Prefer candidates parsed from explicit final-answer phrasing
        candidate = AtomThinkMCTS._extract_answer_candidate(response)
        normalized = _normalize_candidate(candidate)
        if normalized:
            return normalized

        boxed_matches = re.findall(r"\\boxed\{([A-D])\}", response, flags=re.IGNORECASE)
        if boxed_matches:
            return boxed_matches[-1].upper()

        patterns = [
            r"(?:final\s+answer|answer)\s*(?:is|:)\s*([ABCD])\b",
            r"(?:correct|choose|select)\s*(?:option|answer)?\s*([ABCD])\b",
            r"(?:option|choice)\s*([ABCD])\b",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        # Fall back to scanning standalone letters, but prefer the last occurrence
        matches = list(re.finditer(r"\b([ABCD])\b", response, flags=re.IGNORECASE))
        for match in reversed(matches):
            start = match.start()
            end = match.end()
            surrounding = response[max(0, start - 2):end + 2]
            # Skip obvious enumerations like "A, B, C, or D"
            if "," in surrounding:
                continue
            return match.group(1).upper()
        return None

    @staticmethod
    def _unwrap_metadata(metadata):
        if isinstance(metadata, (list, tuple)) and len(metadata) >= 2:
            potential_meta = metadata[1]
            if isinstance(potential_meta, dict):
                return potential_meta
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _get_question_type(self, metadata) -> Optional[str]:
        meta_dict = self._unwrap_metadata(metadata)
        qtype = meta_dict.get("question_type")
        if not qtype and isinstance(meta_dict.get("metadata"), dict):
            qtype = meta_dict["metadata"].get("question_type")
        return qtype

    def _normalize_multi_choice_response(self, response: str, metadata) -> str:
        if not response:
            return response
        question_type = self._get_question_type(metadata)
        if question_type not in {"multi-choice", "multi_choice", "multiple-choice"}:
            return response
        choice = self._extract_choice_from_response(response)
        if not choice:
            return response
        if "final answer" in response.lower():
            return f"To sum up, the final answer is: {choice}."
        return f"The answer is: {choice}."

    def _is_confident_final_answer(self, response: Optional[str], metadata) -> bool:
        if not response:
            return False
        if not has_final_answer_marker(response):
            return False
        candidate = self._extract_answer_candidate(response)
        if not candidate:
            return False
        lowered = candidate.lower()
        if lowered in {"unknown", "none", "n/a", "not sure", "unsure"}:
            return False
        if lowered.startswith(("calculate", "find", "determine", "compute", "show ", "prove ", "explain")):
            return False
        question_type = self._get_question_type(metadata)
        if question_type in {"multi-choice", "multi_choice", "multiple-choice"}:
            choice = self._extract_choice_from_response(candidate)
            if not choice:
                return False
        return True

    def _find_best_terminal_node(self, root_node):
        best_node = None
        stack = [root_node]
        while stack:
            current = stack.pop()
            if current.is_terminal and current.visits > 0:
                if best_node is None:
                    best_node = current
                else:
                    current_score = current.value / current.visits if current.visits else float('-inf')
                    best_score = best_node.value / best_node.visits if best_node.visits else float('-inf')
                    if current_score > best_score:
                        best_node = current
            stack.extend(current.children)
        return best_node


    def _select(self, node, assign_callback=None):
        """
        选择阶段：从根节点开始，递归选择UCT值最高的子节点，直到叶节点
        """
        path = []
        selection_steps: List[Dict[str, Any]] = []
        current = node
        while True:
            if assign_callback is not None:
                assign_callback(current)
            path.append(current)
            if not current.children:
                break
            valid_children = [n for n in current.children if hasattr(n, 'visits')]
            if not valid_children:
                break

            child_records: List[Tuple[MCTSNode, Dict[str, Any]]] = []
            exploration = self.generating_args.mcts_exploration_factor
            for child in valid_children:
                if assign_callback is not None:
                    assign_callback(child)
                uct_val = child.uct(exploration)
                child_records.append(
                    (
                        child,
                        {
                            "node_id": getattr(child, "node_id", None),
                            "depth": len(child.state),
                            "visits": child.visits,
                            "value": child.value,
                            "uct": uct_val,
                            "is_terminal": child.is_terminal,
                        },
                    )
                )

            if not child_records:
                break

            child_records.sort(key=lambda item: item[1]["uct"], reverse=True)
            best_child, best_info = child_records[0]
            reason = "unvisited" if best_child.visits == 0 else "max_uct"

            step_payload = {
                "parent_id": getattr(current, "node_id", None),
                "parent_depth": len(current.state),
                "parent_visits": current.visits,
                "parent_value": current.value,
                "selected_child_id": best_info["node_id"],
                "reason": reason,
                "children": [],
            }
            for rank, (child, info) in enumerate(child_records, start=1):
                info_payload = info.copy()
                info_payload["rank"] = rank
                info_payload["selected"] = info["node_id"] == best_info["node_id"]
                step_payload["children"].append(info_payload)
            selection_steps.append(step_payload)

            current = best_child
        return current, path, selection_steps

    def _expand(self, node, input_item, init_prompt, assign_callback=None):
        """
        扩展阶段：为选定的叶节点生成所有可能的子节点
        """
        created_nodes: List[MCTSNode] = []
        # Only expand if the node is not terminal and hasn't been fully expanded yet
        if node.is_terminal or node.untried_actions is not None:
             # If untried_actions is empty list [], it means fully expanded or sterile
             if node.untried_actions == []: return created_nodes
             # If it's None, we need to generate actions
             pass # Continue to generate actions if None

        candidate_num = self.generating_args.candidate_num
        rollout = node.state
        last_response = rollout[-1] if rollout else ""

        try:
            temp_input_item = update_inputs_from_rollout(copy.deepcopy(input_item), rollout, self.processor)
            # Check length before proceeding
            if len(temp_input_item['inputs'].data['input_ids'][0]) > self.data_args.cutoff_len:
                logger.warning(f"Rollout too long ({len(temp_input_item['inputs'].data['input_ids'][0])} > {self.data_args.cutoff_len}), marking node as terminal.")
                node.is_terminal = True
                node.untried_actions = [] # Mark as fully expanded (sterile)
                return created_nodes
        except Exception as e:
             logger.error(f"Error in update_inputs_from_rollout: {e}")
             node.is_terminal = True # Treat as terminal if update fails
             node.untried_actions = []
             return created_nodes


        node.untried_actions = [] # Initialize list to store potential children
        generated_responses = set()
        temperature = self.generating_args.temperature
        for _ in range(candidate_num):
            try:
                candidate_c = inference_one_step(
                    temp_input_item['inputs'], self.model, self.processor,
                    self.generating_args, temperature
                )
                # 增加温度以获得多样性 (可选，MCTS本身就有探索性)
                # temperature = min(1.0, temperature + 0.5)

                normalized_candidate = candidate_c.strip()
                if normalized_candidate:
                    dedup_key = normalized_candidate
                    if dedup_key in generated_responses:
                        continue
                else:
                    dedup_key = None

                is_valid, is_final_answer = txt_verifier(candidate_c, last_response)
                if is_valid:
                    new_state = rollout + [candidate_c]
                    child_node = MCTSNode(state=new_state, parent=node)
                    # Check if the new step makes it terminal according to verifier
                    child_node.is_terminal = is_final_answer
                    node.untried_actions.append(child_node)
                    created_nodes.append(child_node)
                    if dedup_key is not None:
                        generated_responses.add(dedup_key)
                    if assign_callback is not None:
                        assign_callback(child_node)

                    # 当节点尚未终结时，尝试立即补充一条强制终答，以减少长时间循环
                    if not is_final_answer:
                        forced_response = self._force_final_answer_generation(input_item, new_state)
                        if forced_response and has_final_answer_marker(forced_response):
                            forced_state = new_state + [forced_response]
                            forced_node = MCTSNode(state=forced_state, parent=node)
                            forced_node.is_terminal = True
                            forced_key = forced_response.strip()
                            if forced_key and forced_key not in generated_responses:
                                node.untried_actions.append(forced_node)
                                generated_responses.add(forced_key)
                                created_nodes.append(forced_node)
                                if assign_callback is not None:
                                    assign_callback(forced_node)
            except Exception as e:
                 logger.error(f"Error during inference_one_step in expansion: {e}")
                 # Decide how to handle inference errors, e.g., skip this candidate
        return created_nodes

    def _simulate_and_backpropagate(self, node, init_prompt, assign_callback=None):
        """
        模拟与反向传播阶段：
        1. (模拟) 对新节点进行评估，这里直接使用 reward model 的评分作为模拟结果
        2. (反向传播) 将评分结果沿路径传回至根节点，更新所有父节点的 visits 和 value
        """
        reward = 0.0 # Default reward
        try:
            # 使用 reward model 评估当前状态的分数
            if self.model_args.prm_model_type == "qwen_math_prm":
                reward = self.reward_qwen(init_prompt, node.state)
            else:
                # Assuming a generic self.reward exists for other prm types
                reward = self.reward(init_prompt, node.state)
        except Exception as e:
             logger.error(f"Error during reward calculation for node {node.state}: {e}")
             reward = 0.0 # Assign a default/neutral reward on error


        path_nodes: List[MCTSNode] = []
        temp_node = node
        while temp_node is not None:
            if assign_callback is not None:
                assign_callback(temp_node)
            path_nodes.append(temp_node)
            temp_node = temp_node.parent

        backprop_trace: List[Dict[str, Any]] = []
        for path_node in path_nodes:
            backprop_trace.append(
                {
                    "node": path_node,
                    "node_id": getattr(path_node, "node_id", None),
                    "depth": len(path_node.state),
                    "visits_before": path_node.visits,
                    "value_before": path_node.value,
                }
            )

        for entry in backprop_trace:
            target_node = entry["node"]
            target_node.visits += 1
            target_node.value += reward  # Add the reward obtained at the simulated node
            entry["visits_after"] = target_node.visits
            entry["value_after"] = target_node.value

        for entry in backprop_trace:
            entry.pop("node", None)
        return reward, backprop_trace

    def run_inference(self):
        def _safe_number(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, float):
                if math.isnan(value):
                    return "nan"
                if math.isinf(value):
                    return "inf" if value > 0 else "-inf"
                return value
            if isinstance(value, (int, bool)):
                return value
            return value

        for idx, input_item in enumerate(tqdm(self.inputs)):
            input_item = copy.deepcopy(input_item)
            input_item = self._ensure_final_answer_prompt(input_item)
            metadata_id = None
            if isinstance(self.results, dict):
                metadata_id = input_item.get('metadata', [None, None])[0]
                metadata = input_item.get('metadata', [None, None])[1]
                if metadata is None:
                    metadata = {}
            else:
                metadata = input_item.get('metadata', {})
                if metadata is None:
                    metadata = {}

            sample_index = input_item.get("sample_index")
            sample_index_str = str(sample_index) if sample_index is not None else None
            if sample_index_str is None and isinstance(metadata_id, (str, int)):
                sample_index_str = str(metadata_id)

            start_time = time.time()
            init_prompt = copy.deepcopy(input_item.get("prompt", ""))
            root = MCTSNode(state=[])

            trace_active = self._should_trace_sample(sample_index_str, idx)
            trace_state = None
            if trace_active:
                trace_state = {"nodes": {}, "next_id": 0, "events": []}
                trace_assign = lambda node: self._trace_assign_node(trace_state, node)
            else:
                trace_assign = lambda node: None
            trace_assign(root)

            num_iterations = self.generating_args.mcts_iterations
            resolved_via_fail_try = False
            early_stop_node = None
            early_stop_reason = None
            chosen_node = None

            for iter_num in range(num_iterations):
                leaf_node, selection_path, selection_steps = self._select(root, trace_assign)

                created_nodes = []
                if not leaf_node.is_terminal:
                    created_nodes = self._expand(leaf_node, input_item, init_prompt, trace_assign) or []

                node_to_simulate = leaf_node
                if leaf_node.untried_actions:
                    node_to_simulate = leaf_node.untried_actions.pop(
                        random.randrange(len(leaf_node.untried_actions))
                    )
                    leaf_node.children.append(node_to_simulate)

                trace_assign(node_to_simulate)
                reward, backprop_trace = self._simulate_and_backpropagate(
                    node_to_simulate, init_prompt, trace_assign
                )

                if trace_active and trace_state is not None:
                    serialized_selection = []
                    for step in selection_steps:
                        children_info = []
                        for child in step.get("children", []):
                            children_info.append(
                                {
                                    "node_id": child.get("node_id"),
                                    "depth": child.get("depth"),
                                    "visits": child.get("visits"),
                                    "value": child.get("value"),
                                    "uct": _safe_number(child.get("uct")),
                                    "is_terminal": child.get("is_terminal"),
                                    "rank": child.get("rank"),
                                    "selected": child.get("selected", False),
                                }
                            )
                        serialized_selection.append(
                            {
                                "parent_id": step.get("parent_id"),
                                "parent_depth": step.get("parent_depth"),
                                "parent_visits": step.get("parent_visits"),
                                "parent_value": step.get("parent_value"),
                                "selected_child_id": step.get("selected_child_id"),
                                "reason": step.get("reason"),
                                "children": children_info,
                            }
                        )

                    serialized_backprop = [
                        {
                            "node_id": entry.get("node_id"),
                            "depth": entry.get("depth"),
                            "visits_before": entry.get("visits_before"),
                            "visits_after": entry.get("visits_after"),
                            "value_before": entry.get("value_before"),
                            "value_after": entry.get("value_after"),
                        }
                        for entry in backprop_trace
                    ]

                    event = {
                        "iteration": iter_num,
                        "selected_path": [node.node_id for node in selection_path if node.node_id is not None],
                        "expanded_nodes": [child.node_id for child in created_nodes if child.node_id is not None],
                        "simulated_node": node_to_simulate.node_id,
                        "reward": reward,
                        "node_visits": node_to_simulate.visits,
                        "node_value": node_to_simulate.value,
                        "is_terminal": node_to_simulate.is_terminal,
                        "selection_steps": serialized_selection,
                        "backprop": serialized_backprop,
                    }
                else:
                    event = None

                if (
                    node_to_simulate.is_terminal
                    and node_to_simulate.state
                    and self._is_confident_final_answer(node_to_simulate.state[-1], metadata)
                ):
                    early_stop_node = node_to_simulate
                    early_stop_reason = "confident_terminal"
                    if trace_active and trace_state is not None and event is not None:
                        event["early_stop"] = True
                        trace_state["events"].append(event)
                    break

                if trace_active and trace_state is not None and event is not None:
                    trace_state["events"].append(event)

            response = None
            best_rollout: List[str] = []
            selected_visits = 0
            if early_stop_node and early_stop_node.state:
                chosen_node = early_stop_node
                best_rollout = early_stop_node.state
                response = best_rollout[-1]
                selected_visits = early_stop_node.visits
            else:
                best_terminal_node = self._find_best_terminal_node(root)
                if best_terminal_node and best_terminal_node.state:
                    chosen_node = best_terminal_node
                    best_rollout = best_terminal_node.state
                    response = best_rollout[-1]
                    selected_visits = best_terminal_node.visits
                else:
                    best_node = root
                    while best_node.children:
                        valid_children = [n for n in best_node.children if hasattr(n, 'visits')]
                        if not valid_children:
                            break
                        best_node = max(valid_children, key=lambda n: n.visits)
                    if best_node.state:
                        chosen_node = best_node
                        best_rollout = best_node.state
                        response = best_rollout[-1]
                        selected_visits = best_node.visits

            if response is not None:
                if not self._is_confident_final_answer(response, metadata):
                    logger.info(
                        f"Best path ({selected_visits} visits) did not yield a reliable final answer. Refining the conclusion."
                    )
                    wrapped_response = self._wrap_with_final_answer(best_rollout)
                    if wrapped_response and self._is_confident_final_answer(wrapped_response, metadata):
                        response = wrapped_response
                    else:
                        forced_response = self._force_final_answer_generation(input_item, best_rollout)
                        if forced_response and self._is_confident_final_answer(forced_response, metadata):
                            response = forced_response
                        else:
                            logger.info("Forced final answer failed; falling back to fail_try().")
                            response = self.fail_try(input_item, best_rollout)
                            resolved_via_fail_try = True
            else:
                logger.warning("MCTS finished without finding a valid path. Trying fail_try().")
                response = self.fail_try(input_item, best_rollout)
                resolved_via_fail_try = True

            response = self._normalize_multi_choice_response(response, metadata)

            top_n_to_collect = 5
            top_trajectories = self._collect_top_trajectories(root, top_n=top_n_to_collect)

            end_time = time.time()
            metadata["response"] = response
            metadata["prompt"] = init_prompt
            metadata["steps"] = [best_rollout]
            metadata["total_inference_times"] = root.visits
            metadata["max_mcts_iterations"] = num_iterations
            metadata["resolved_via_fail_try"] = resolved_via_fail_try
            if early_stop_reason:
                metadata["early_stop_reason"] = early_stop_reason
            metadata["mcts_top_trajectories"] = top_trajectories

            if isinstance(self.results, dict):
                if metadata_id is not None:
                    self.results[metadata_id] = metadata
                else:
                    logger.warning("Missing metadata_id, cannot save result to dict.")
            elif isinstance(self.results, list):
                self.results.append(metadata)
            else:
                logger.error("self.results is neither list nor dict.")

            if trace_active and trace_state is not None:
                trace_path = self._resolve_trace_path(sample_index_str, idx)
                if chosen_node is not None:
                    self._trace_assign_node(trace_state, chosen_node)
                sample_payload = {
                    "sample_index": sample_index_str,
                    "dataset_position": idx,
                    "question": input_item.get("question"),
                    "question_type": input_item.get("question_type"),
                    "answer": input_item.get("answer"),
                    "response": response,
                    "best_rollout": best_rollout,
                    "best_node_id": chosen_node.node_id if chosen_node else None,
                    "best_path_ids": self._trace_path_ids(trace_state, chosen_node),
                    "resolved_via_fail_try": resolved_via_fail_try,
                    "early_stop_reason": early_stop_reason,
                    "selected_visits": selected_visits,
                    "total_visits": root.visits,
                    "max_iterations": num_iterations,
                }
                if isinstance(metadata, dict):
                    sample_payload["metadata_subject"] = metadata.get("subject")
                    sample_payload["metadata_source"] = metadata.get("source")
                self._dump_trace_output(trace_state, root, trace_path, sample_payload)

            logger.info(
                f"Time: {end_time - start_time:.4f} | Total MCTS Iterations: {num_iterations}"
            )
            save_json(self.data_args.answers_file, self.results)


    def fail_try(self, input_item, best_rollout=None):
        question = input_item.get('prompt', '').split("THE GIVEN QUESTION:\n")[-1].split("HISTORICAL REASONING STEPS:")[
            0].replace("\n", " ").strip()
        question += "\nAnswer the question using a single word or phrase."
        question_type = self._get_question_type(input_item.get('metadata'))
        if question_type in {"multi-choice", "multi_choice", "multiple-choice"}:
            question += "\nWhen you give the final answer, respond with only the correct option letter (A, B, C, or D)."
        if best_rollout:
            joined_steps = "\n".join(step.strip() for step in best_rollout if step.strip())
            if joined_steps:
                question += "\nHere is the reasoning collected so far:\n" + joined_steps
        question += "\nPlease respond in the form \"The answer is: <final_answer>.\""

        # Ensure necessary components are available
        template = getattr(self.data_args, 'template', None)
        processor = getattr(self, 'processor', None)
        tokenizer = getattr(self, 'tokenizer', None)
        model_name = getattr(self.model_args, 'model_name_or_path', None)
        images = input_item.get('images', None)

        if not all([template, processor, tokenizer, model_name]):
             logger.error("Missing necessary components (template, processor, tokenizer, model_name) for fail_try.")
             return "Error: fail_try configuration incomplete."

        try:
            input_prompt = conversation_map[template](question,
                                                     "image", # Assuming "image" is a placeholder type
                                                     model_name_or_path=model_name,
                                                     processor=processor,
                                                     tokenizer=tokenizer,
                                                     images=images
                                                     )
            logger.info(f"fail_try() prompt: {input_prompt}") # Log the specific prompt used
            input_ids = processor(
                text=[input_prompt],
                images=images,
                padding=True,
                return_tensors="pt",
            ).to("cuda") # Assuming you want to run this on cuda
            response = inference_one_step(input_ids,
                                          self.model,
                                          processor,
                                          self.generating_args,
                                          self.generating_args.temperature # Using main temperature here
                                          )
        except Exception as e:
            logger.error(f"Error during fail_try inference: {e}")
            response = "Error during fail_try inference." # Return error message

        # response 此时可能是 "C" 或 "7" 这样的裸奔答案
        # 我们把它包装成评分脚本(eval_utils.py)认识的格式
        naked_answer = re.sub(r"\s+", " ", response.strip())
        if not naked_answer:
            naked_answer = "unknown"
        if re.search(r"^\s*(the answer is|to sum up, the final answer is)", naked_answer, flags=re.IGNORECASE):
            formatted_response = naked_answer
            if not formatted_response.endswith("."):
                formatted_response += "."
        else:
            trimmed = naked_answer.rstrip(".")
            formatted_response = f"The answer is: {trimmed}."

        formatted_response = self._normalize_multi_choice_response(formatted_response, input_item.get('metadata'))
        logger.info(f"fail_try() wrapped naked answer '{naked_answer}' into '{formatted_response}'")
        return formatted_response
        # --- 【修复 2 / 最终修复 v2 - 保留】 结束 ---

    @staticmethod
    def load_reward_model(model_args):
        # Ensure prm_model attribute exists
        prm_model_pretrained = getattr(model_args, 'prm_model', None)
        if not prm_model_pretrained:
            logger.error("Reward model path (prm_model) not specified in model_args.")
            # Return None or raise error depending on whether reward model is optional
            return None, None, None, None # Or raise ValueError("prm_model path is required")

        logger.info(f"Loading reward model from: {prm_model_pretrained}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                prm_model_pretrained,
                add_eos_token=False,
                trust_remote_code=getattr(model_args, 'trust_remote_code', True) # Add trust_remote_code
                )
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id # Common practice
            tokenizer.padding_side = "left" # Important for generation

            prm_model_type = getattr(model_args, 'prm_model_type', '')
            prm_device_map = getattr(model_args, 'prm_device_map', 'auto')
            dtype_str = getattr(model_args, 'infer_dtype', 'auto')
            dtype = getattr(torch, dtype_str) if dtype_str != 'auto' else torch.bfloat16 # Default bfloat16

            model = None
            candidate_tokens = None
            step_tag_id = None

            if prm_model_type == "qwen_math_prm":
                candidate_tokens = [None, None]
                step_tag_id = None
                # Use AutoModel for models needing trust_remote_code=True for custom architectures
                model = AutoModel.from_pretrained(
                    prm_model_pretrained,
                    device_map=prm_device_map,
                    torch_dtype=dtype,
                    trust_remote_code=getattr(model_args, 'trust_remote_code', True),
                ).eval()
            else:
                # Default for standard Causal LM reward models
                good_token = '+'
                bad_token = '-'
                step_tag = '\n\n\n\n\n' # Example step tag, adjust if needed
                try:
                     # Ensure encoding uses appropriate format, handle potential errors
                     candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}", add_special_tokens=False)
                     step_tag_tokens = tokenizer.encode(f" {step_tag}", add_special_tokens=False)
                     step_tag_id = step_tag_tokens[-1] if step_tag_tokens else None
                except Exception as e:
                     logger.error(f"Error encoding special tokens for reward model: {e}")
                     candidate_tokens = []
                     step_tag_id = None


                model = AutoModelForCausalLM.from_pretrained(
                    prm_model_pretrained,
                    device_map=prm_device_map,
                    torch_dtype=dtype,
                    trust_remote_code=getattr(model_args, 'trust_remote_code', True)
                    ).eval() # Put model in eval mode

            logger.info("Reward model loaded successfully.")
            return model, tokenizer, candidate_tokens, step_tag_id

        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            # Depending on requirements, either raise the error or return None
            # raise e
            return None, None, None, None


    def reward_qwen(self, init_prompt, rollout, caption=None, ):
        # Ensure prm_model and tokenizer are loaded
        if not self.prm_model or not self.prm_tokenizer:
             logger.error("Reward model or tokenizer not loaded, cannot calculate reward.")
             return 0.0 # Return default/neutral reward

        # Extract question safely
        try:
            question_part = init_prompt.split("THE GIVEN QUESTION:\n", 1)[1]
            question = question_part.split("HISTORICAL REASONING STEPS:", 1)[0].replace("\n", " ").strip()
        except IndexError:
             logger.warning("Could not parse question from init_prompt using expected format.")
             question = init_prompt # Use the whole prompt as fallback

        if caption:
            question = question + " " + caption

        system = "Please reason step by step, and put your final answer within \\boxed{}."
        assistant_content = "<extra_0>".join(rollout) + "<extra_0>"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]

        try:
            conversation_str = self.prm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            input_ids = self.prm_tokenizer.encode(
                conversation_str,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.prm_device)

            with torch.no_grad():
                outputs = self.prm_model(input_ids=input_ids)

            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)
            probabilities = F.softmax(logits.float(), dim=-1)

            sep_ids = self.prm_tokenizer.encode("<extra_0>", add_special_tokens=False)
            if not sep_ids:
                logger.error("Could not encode step separator token: <extra_0>")
                return 0.0
            step_sep_id = sep_ids[0]

            token_mask = (input_ids == step_sep_id)
            if token_mask.dim() == 1:
                token_mask = token_mask.unsqueeze(0)

            masked_probs = probabilities[token_mask]
            if masked_probs.numel() == 0:
                fallback_logits = logits[:, -1, :]
                fallback_probs = F.softmax(fallback_logits, dim=-1)
                positive_prob = fallback_probs[:, 1].item() if fallback_probs.shape[-1] > 1 else 0.0
                res = [positive_prob]
            else:
                if masked_probs.dim() == 1:
                    masked_probs = masked_probs.unsqueeze(0)
                if masked_probs.shape[-1] == 2:
                    res = masked_probs[:, 1].cpu().tolist()
                else:
                    logger.warning(f"Unexpected output dimension ({masked_probs.shape[-1]}) at separator tokens. Expected 2.")
                    fallback_logits = logits[:, -1, :]
                    fallback_probs = F.softmax(fallback_logits, dim=-1)
                    positive_prob = fallback_probs[:, 1].item() if fallback_probs.shape[-1] > 1 else 0.0
                    res = [positive_prob]

            if not res:
                logger.warning("Reward calculation resulted in empty score list.")
                return 0.0

            aggregation_method = getattr(self.generating_args, 'aggregation', 'last')
            if aggregation_method == "avg":
                return float(sum(res)) / len(res)
            elif aggregation_method == "min":
                return float(min(res))
            else:
                return float(res[-1])

        except Exception as e:
            logger.error(f"Exception during reward_qwen calculation: {e}")
            return 0.0 # Return default/neutral reward on error

    def reward(self, init_prompt, rollout, caption=None):
        question = self._extract_question_text(init_prompt)
        if caption:
            question = f"{question}\n{caption}"
        reasoning = self._format_rollout(rollout)
        prompt = (
            "You are a process reward model for mathematical reasoning. "
            "Given the following question and reasoning steps, respond with '+' if the reasoning is correct "
            "or '-' if it is flawed.\n\n"
            f"Question:\n{question.strip()}\n\nReasoning:\n{reasoning.strip()}\n\nAnswer:"
        )

        base_device = self._resolve_prm_device(self.prm_model)
        inputs = self.prm_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(base_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(base_device)

        text_model = getattr(self.prm_model, "language_model", self.prm_model)
        with torch.no_grad():
            outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        plus_id, minus_id = (self.candidate_tokens or [None, None])
        plus_prob = probs[0, plus_id].item() if plus_id is not None else 0.0
        minus_prob = probs[0, minus_id].item() if minus_id is not None else 0.0
        total = plus_prob + minus_prob
        if total <= 0:
            return plus_prob
        return plus_prob / total

    @staticmethod
    def _extract_question_text(init_prompt: str) -> str:
        if not init_prompt:
            return ""
        marker = "THE GIVEN QUESTION:\n"
        if marker in init_prompt:
            tail = init_prompt.split(marker, 1)[1]
            if "HISTORICAL REASONING STEPS:" in tail:
                tail = tail.split("HISTORICAL REASONING STEPS:", 1)[0]
            return tail.strip()
        return init_prompt.strip()

    @staticmethod
    def _format_rollout(rollout):
        if not rollout:
            return "No reasoning provided."
        lines = []
        for idx, step in enumerate(rollout, 1):
            normalized = (step or "").strip()
            lines.append(f"Step {idx}: {normalized}")
        return "\n".join(lines)

    @staticmethod
    def _resolve_prm_device(model: Optional[torch.nn.Module]) -> torch.device:
        if model is None:
            return torch.device('cpu')
        if hasattr(model, 'device'):
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')
