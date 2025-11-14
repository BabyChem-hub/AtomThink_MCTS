import json
import os
import threading
from tqdm import tqdm
import time
import math
from llamafactory.evaluation.utils.prompts import demo_prompt_score
from openai import OpenAI
from loguru import logger
import re
from os.path import join, isdir, isfile, isdir, dirname
import yaml
import time

try:
    import sympy
except ImportError:
    sympy = None

MAX_TRY = 5


class LocalLLMJudge:
    def __init__(self, model_path, device_map="auto", torch_dtype="bfloat16", temperature=0.0,
                 max_new_tokens=32, system_prompt=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM judge model path does not exist: {model_path}")

        self.temperature = max(float(temperature), 0.0)
        self.do_sample = self.temperature > 0.0
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.system_prompt = system_prompt or (
            "You are a strict math evaluation assistant. Reply with 1 when answers match exactly; otherwise reply with 0."
        )

        tokenizer_kwargs = {"trust_remote_code": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

        model_kwargs = {"device_map": device_map, "trust_remote_code": True}
        dtype = self._resolve_dtype(torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except ValueError as exc:
            if "Qwen2_5_VLConfig" in str(exc) or "Qwen2_5_VL" in str(exc):
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            else:
                raise
        self.model.eval()
        self.device = self._infer_device()
        self.input_device = self._infer_input_device()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._can_use_chat_template = True

    @staticmethod
    def _resolve_dtype(dtype_name):
        import torch
        if not dtype_name or dtype_name == "auto":
            return None
        lookup = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype = lookup.get(str(dtype_name).lower())
        if dtype is None:
            logger.warning(f"Unsupported torch dtype {dtype_name}, defaulting to auto")
        return dtype


    def _infer_device(self):
        import torch

        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            def normalize(device):
                if isinstance(device, (list, tuple)) and device:
                    return device[0]
                return device

            devices = [normalize(device) for device in device_map.values() if device is not None]
            for candidate in devices:
                if candidate is None:
                    continue
                if str(candidate).startswith("cuda:0"):
                    try:
                        return torch.device(candidate)
                    except Exception:
                        continue
            for candidate in devices:
                if candidate is None:
                    continue
                try:
                    return torch.device(candidate)
                except Exception:
                    continue
        if hasattr(self.model, "device") and self.model.device is not None:
            device = self.model.device
            if isinstance(device, torch.device):
                return device
            try:
                return torch.device(device)
            except Exception:
                pass
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _infer_input_device(self):
        import torch

        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            def normalize(device):
                if isinstance(device, (list, tuple)) and device:
                    return device[0]
                return device

            preferred_keys = (
                "embed_tokens",
                "input_block",
                "feature_extractor",
                "model.0",
            )
            for key in preferred_keys:
                for module, device in device_map.items():
                    if key in module:
                        device_str = normalize(device)
                        try:
                            return torch.device(device_str)
                        except Exception:
                            continue
            first_device = normalize(next(iter(device_map.values())))
            try:
                return torch.device(first_device)
            except Exception:
                pass
        return self.device

    @staticmethod
    def _parse_output(text):
        match = re.search(r'(?<!\d)([01])(?!\d)', text)
        if match:
            return int(match.group(1))
        return None

    def judge(self, question, model_answer, standard_answer):
        import torch

        question = "" if question is None else str(question)
        model_answer = "" if model_answer is None else str(model_answer)
        standard_answer = "" if standard_answer is None else str(standard_answer)

        prompt = demo_prompt_score.format(question=question, gt=standard_answer, extraction=model_answer)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        prompt_text = None
        if self._can_use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                logger.warning(f"Failed to use chat template: {exc}. Falling back to manual prompt formatting.")
                self._can_use_chat_template = False
        if prompt_text is None:
            prompt_text = ""
            if self.system_prompt:
                prompt_text += (
                    "<|im_start|>system\n"
                    f"{self.system_prompt}\n"
                    "<|im_end|>\n"
                )
            prompt_text += (
                "<|im_start|>user\n"
                f"{prompt}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        generated = outputs[:, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        judgement = self._parse_output(text)
        return judgement, text


_LLM_JUDGE_INSTANCE = None
_LLM_JUDGE_LOCK = threading.Lock()


def _flag_enabled(value, default="0"):
    if value is None:
        value = default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {name}: {value}, falling back to {default}")
        return default


def _get_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid int for {name}: {value}, falling back to {default}")
        return default


def _llm_judge_enabled():
    return _flag_enabled(os.getenv("USE_LLM_JUDGE"))


def _get_llm_judge():
    global _LLM_JUDGE_INSTANCE
    if not _llm_judge_enabled():
        return None
    if _LLM_JUDGE_INSTANCE is not None:
        return _LLM_JUDGE_INSTANCE
    with _LLM_JUDGE_LOCK:
        if _LLM_JUDGE_INSTANCE is not None:
            return _LLM_JUDGE_INSTANCE
        config = {
            "model_path": os.getenv("LLM_JUDGE_MODEL_PATH", "/data1/xiangkun/MODELS/Qwen2.5-VL-7B-Instruct"),
            "device_map": os.getenv("LLM_JUDGE_DEVICE_MAP", "auto"),
            "torch_dtype": os.getenv("LLM_JUDGE_DTYPE", "bfloat16"),
            "temperature": _get_env_float("LLM_JUDGE_TEMPERATURE", 0.0),
            "max_new_tokens": _get_env_int("LLM_JUDGE_MAX_NEW_TOKENS", 32),
            "system_prompt": os.getenv(
                "LLM_JUDGE_SYSTEM_PROMPT",
                "You are a strict math evaluation assistant. Reply with 1 when answers match exactly; otherwise reply with 0.",
            ),
        }
        try:
            _LLM_JUDGE_INSTANCE = LocalLLMJudge(**config)
        except Exception as exc:
            logger.error(f"Failed to initialize local LLM judge: {exc}")
            _LLM_JUDGE_INSTANCE = None
    return _LLM_JUDGE_INSTANCE

def s0_merge_answers(model_output_dir, dataset_name, tag):
    inference_dir = join(model_output_dir, 'inference', dataset_name)
    answers_file = join(inference_dir, f'answers_{tag}.json')

    subfolders = []
    for entry in os.listdir(inference_dir):
        entry_path = os.path.join(inference_dir, entry)
        if os.path.isdir(entry_path) and 'tmp' not in entry_path:
            subfolders.append(entry_path)
    ans_list = []
    ans_dict = {}
    for subfolder in subfolders:
        for filename in os.listdir(subfolder):
            file = os.path.join(subfolder, filename)
            if os.path.isfile(file) and tag in filename:
                answers = read_json(file)
                if isinstance(answers, dict):
                    ans_dict = {**ans_dict, **answers}
                else:
                    ans_list.extend(answers)
    if ans_dict:
        save_json(answers_file, ans_dict)
        print(f'Total Answers Length is: {len(ans_dict)}')
        time.sleep(5)
    else:
        save_json(answers_file, ans_list)
        print(f'Total Answers Length is: {len(ans_list)}')
        time.sleep(5)
    return answers_file

def s1_separate_n(answers_file, n_process):
    split_dir = dirname(answers_file)
    split_dir = join(split_dir, 'tmp')
    if not isdir(split_dir):
        os.makedirs(split_dir)
    total_answers = read_json(answers_file)
    if isinstance(total_answers, dict):
        total_answers = list(total_answers.items())
        output_type = "dict"
    else:
        output_type = "list"
    answers_file_list = []
    total_length = len(total_answers)
    chunk_size = total_length // n_process
    remainder = total_length % n_process

    start = 0
    for i in range(n_process):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = total_answers[start:end]

        if output_type == "dict":
            chunk = dict(chunk)
        filename = f"{i}.json"
        filename = join(split_dir, filename)
        save_json(filename, chunk)
        answers_file_list.append(filename)
        print(f"Saved {filename} with {len(chunk)} elements.")
        start = end
    return answers_file_list

def s3_merge_save_answers(answers_file, answers_file_list):
    ans_list = []
    ans_dict = {}
    for file in answers_file_list:
        try:
            if '.jsonl' in file:
                answers = read_json(file)
            else:
                answers = read_json(file)
        except Exception as e:
            print(f'error: {e}\n{file}')
            continue
        if isinstance(answers, dict):
            ans_dict = {**ans_dict, **answers}
        else:
            ans_list.extend(answers)
    if isinstance(answers, dict):
        save_json(answers_file, ans_dict)
    else:
        save_json(answers_file, ans_list)

def save_jsonl(path: str, data: list, ) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for line in tqdm(data, desc='save'):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(path: str, key: str = None):
    data = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if not line:
                continue
            data.append(json.loads(line))

    if key is not None:
        data.sort(key=lambda x: x[key])
        data = {item[key]: item for item in data}
    return data

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict

def save_yaml(data, save_path):
    with open(save_path, "w") as f:
        yaml.dump(data, f)

def score_func(response, query, gt):
    if response is None or str(response).strip() == "":
        return 0

    pred = str(response).strip()
    target = str(gt).strip()

    judge = _get_llm_judge()
    if judge is not None:
        try:
            decision, raw_output = judge.judge(query, pred, target)
            if decision is not None:
                return decision
            logger.warning(
                f"LLM judge returned undecidable result\nProblem: {query}\nPred: {pred}\nGT: {target}\nModel Output: {raw_output}"
            )
        except Exception as exc:
            logger.error(f"LLM judge failed: {exc}")

    def extract_choice(s):
        match = re.match(r'^\s*\(?\s*([A-Da-d])\s*\)?\s*$', s)
        if match:
            return match.group(1).upper()
        return None

    pred_choice = extract_choice(pred)
    target_choice = extract_choice(target)
    if pred_choice and target_choice:
        return int(pred_choice == target_choice)

    def normalize(text):
        if text is None:
            return ""
        text = str(text).strip()
        text = text.lower()
        text = text.replace("π", "pi")
        replacements = {
            r"\\pi": "pi",
            r"\\times": "*",
            r"\\cdot": "*",
            r"\\div": "/",
            r"\\pm": "+/-",
        }
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text)
        text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", text)
        text = re.sub(r"\\left|\\right", "", text)
        text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\operatorname\s*\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\^\s*\{([^}]*)\}", r"**(\1)", text)
        text = re.sub(r"\^([0-9]+)", r"**\1", text)
        text = text.replace("{", "").replace("}", "")
        text = text.replace("$", "")
        text = text.replace("~", "")
        descriptors = [
            "area", "volume", "perimeter", "circumference", "answer",
            "result", "value", "probability", "ratio", "equals"
        ]
        for desc in descriptors:
            text = re.sub(rf"\b{desc}\b[:\s]*", "", text)
        units = [
            "centimeter", "centimeters", "cm", "meter", "meters", "m", "millimeter", "millimeters", "mm",
            "kilometer", "kilometers", "km", "inch", "inches", "in", "foot", "feet", "ft",
            "yard", "yards", "mile", "miles", "kg", "g", "mg", "usd", "dollar", "dollars",
            "percent", "%", "°", "degrees", "cubic", "square", "sq", "unit", "units",
            "hour", "hours", "minute", "minutes", "second", "seconds"
        ]
        for unit in units:
            text = re.sub(rf"\b{unit}\b(?:\*\*\(?-?\d+\)?)?", "", text)
        text = re.sub(r'\b([a-z])\s*=', '', text, flags=re.IGNORECASE)
        text = re.sub(r"[=:\u2260]", " ", text)
        text = re.sub(r"[,\s]+", " ", text).strip()
        text = text.replace(" ", "")
        text = text.rstrip("\\")
        text = re.sub(r"(\d)(pi)", r"\1*pi", text)
        text = re.sub(r"(pi)(\d)", r"pi*\2", text)
        text = re.sub(r"\)(\d)", r")*\1", text)
        text = re.sub(r"(\d)\(", r"\1*(", text)
        text = text.replace("^", "**")
        return text

    def expressions_equal(a, b):
        if a == "" or b == "":
            return False
        if a == b:
            return True
        if sympy is not None:
            try:
                expr_a = sympy.simplify(sympy.sympify(a))
                expr_b = sympy.simplify(sympy.sympify(b))
                if sympy.simplify(expr_a - expr_b) == 0:
                    return True
            except Exception:
                pass
        safe_globals = {"__builtins__": None, "pi": math.pi, "sqrt": math.sqrt}
        try:
            val_a = eval(a, safe_globals, {})
            val_b = eval(b, safe_globals, {})
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                if math.isclose(float(val_a), float(val_b), rel_tol=1e-4, abs_tol=1e-2):
                    return True
        except Exception:
            pass
        return False

    normalized_pred = normalize(pred)
    normalized_target = normalize(target)

    if expressions_equal(normalized_pred, normalized_target):
        return 1

    logger.warning(
        f"Mismatch after normalization!\nProblem: {query}\nPred: {pred}\nGT: {target}\nNorm Pred: {normalized_pred}\nNorm GT: {normalized_target}"
    )
    return 0

def make_api_call(messages, max_tokens=200, temperature=0.2, is_json=False, custom_client=None, model='gpt-4o'):
    if custom_client != None:
        client = custom_client
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", None)
        )
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            client.base_url = base_url
    for attempt in range(5):
        try:
            if not is_json:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 4:
                return f"Error: {str(e)}"
            time.sleep(1)  # Wait for 1 second before retrying

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt

def extract_by_rule(response):
    if response is None:
        return ""
    response_str = str(response).strip()

    def clean_candidate(text):
        text = text.strip()
        if not text:
            return text
        text = re.sub(r'\\text\s*\{[^}]*\}', '', text)
        text = re.sub(r'\\mathrm\s*\{[^}]*\}', '', text)
        text = re.sub(r'\\operatorname\s*\{[^}]*\}', '', text)
        text = re.sub(r'\\left|\\right', '', text)
        text = text.replace('$', '')
        text = text.replace('~', ' ')
        text = text.replace('\\,', ' ')
        if '=' in text:
            text = text.split('=')[-1]
        text = re.sub(r'\b(area|volume|perimeter|circumference|answer|result|value|probability|ratio)\b[:\s]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(cm|m|mm|km|meter|meters|centimeter|centimeters|degree|degrees|unit|units|square|sq|cubic)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s\^\d+', '', text)
        text = re.sub(r'[,:;\[\]\(\)]$', '', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        if text.endswith('.'):
            text = text[:-1].strip()
        return text

    def is_math_expr(text):
        if not text:
            return False
        if re.search(r'\d', text):
            return True
        if re.search(r'\\pi|pi|\\frac|frac|\\sqrt|sqrt', text, re.IGNORECASE):
            return True
        return False

    if '\\boxed{' in response_str:
        start = response_str.find('\\boxed{') + len('\\boxed{')
        depth = 1
        end = start
        while end < len(response_str) and depth > 0:
            if response_str[end] == '{':
                depth += 1
            elif response_str[end] == '}':
                depth -= 1
            end += 1
        candidate = response_str[start:end-1] if depth == 0 else response_str[start:]
        candidate = clean_candidate(candidate)
        if candidate:
            return candidate

    regex_checks = [
        (r'<answer>\s*(.*?)\s*</answer>', re.IGNORECASE | re.DOTALL),
        (r'(?:the\s+final\s+answer\s+is|final\s+answer|the\s+answer\s+is|answer\s*[:\s])\s*([^\n\.]+)', re.IGNORECASE),
    ]
    for pattern, flags in regex_checks:
        try:
            match = re.search(pattern, response_str, flags)
            if match:
                candidate = clean_candidate(match.group(1))
                if candidate:
                    return candidate
        except Exception:
            continue

    eq_candidates = []
    try:
        eq_candidates = re.findall(r'=\s*([^\n\r]+)', response_str)
    except Exception:
        eq_candidates = []
    cleaned_eq = [clean_candidate(cand) for cand in eq_candidates]
    cleaned_eq = [cand for cand in cleaned_eq if is_math_expr(cand)]
    if cleaned_eq:
        return cleaned_eq[-1]

    lines = [clean_candidate(line) for line in response_str.splitlines()]
    for line in reversed(lines):
        if is_math_expr(line):
            return line

    try:
        return str(float(response_str))
    except Exception:
        pass
    return response_str
