from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from transformers import GenerationConfig # Added this missing import for the to_dict method


@dataclass
class GeneratingArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """

    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."},
    )
    temperature: float = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.7,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: int = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Default system message to use in chat completion."},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether or not to remove special tokens in the decoding."},
    )
    max_sampling_count: int = field(
        default=1,
        metadata={"help": "max_sampling_count"},
    )
    max_single_step_sampling_count: int = field(
        default=1,
        metadata={"help": "max_single_step_sampling_count"},
    )
    max_depth: int = field(
        default=1,
        metadata={"help": "max_depth"},
    )
    method: str = field(
        default=None,
        metadata={"help": "method"},
    )
    aggregation: str = field(
        default=None,
        metadata={"help": "aggregation"},
    )
    atomthink_beam_search_num: int = field(
        default=0,
        metadata={"help": "atomthink_beam_search_num"},
    )
    candidate_num: int = field(
        default=0,
        metadata={"help": "candidate_num"},
    )

    # --- START: Added MCTS Parameters ---
    mcts_iterations: int = field(
        default=100,
        metadata={"help": "Number of iterations for the MCTS algorithm."}
    )
    mcts_exploration_factor: float = field(
        default=1.414,
        metadata={"help": "Exploration factor (C) for the UCT formula in MCTS."}
    )
    mcts_exploration_const: Optional[float] = field(
        default=None,
        metadata={"help": "Alias for mcts_exploration_factor to maintain config compatibility."}
    )
    mcts_trace_samples: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma/space separated sample indices (or positional indices) to dump detailed MCTS traces."
        },
    )
    mcts_trace_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save detailed MCTS trace dumps (JSON)."},
    )
    # --- END: Added MCTS Parameters ---

    def __post_init__(self) -> None:
        if self.mcts_exploration_const is not None:
            self.mcts_exploration_factor = self.mcts_exploration_const

    def to_dict(self, obey_generation_config: bool = False) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)

        if obey_generation_config:
            generation_config = GenerationConfig()
            for key in list(args.keys()):
                if not hasattr(generation_config, key):
                    args.pop(key)
        return args
