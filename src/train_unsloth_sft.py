"""Continue supervised fine-tuning a Qwen2.5-Coder model with Unsloth.

The defaults use this project's public Hugging Face model and dataset. Training
produces a LoRA adapter locally; optionally, it can also save or push a merged
16-bit model for normal Transformers/vLLM inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Chimanwakis/qwen_manim_animation_16bit"
DEFAULT_DATASET_ID = "Chimanwakis/calculus_manim"
DEFAULT_DATASET_FILE = "calcus_final_training_dataset_17250.jsonl"
DEFAULT_OUTPUT_DIR = Path("outputs/qwen-calculus-sft")
SEED = 3407

CALCULUS_SYSTEM_PROMPT = (
    "You are an expert calculus educator and Manim Community Edition developer. "
    "Follow the requested task and output format exactly. Preserve all mathematical "
    "conditions, notation, reasoning, and conclusions supplied in the input."
)

CALCULUS_TASK_INSTRUCTIONS = {
    "question_to_solution": (
        "Solve the following calculus problem accurately. Return only a clear, "
        "mathematically correct worked solution; do not produce a storyboard or code."
    ),
    "question_to_storyboard": (
        "Turn the following calculus problem into a mathematically correct, "
        "implementation-ready numbered animation storyboard. Return only the storyboard."
    ),
    "solution_to_storyboard": (
        "Turn the following worked calculus solution into a faithful, self-contained, "
        "implementation-ready numbered animation storyboard. Return only the storyboard."
    ),
    "storyboard_to_manim_code": (
        "Convert the following storyboard into one complete executable Manim Community "
        "Edition Python scene. Return Python code only, beginning with "
        "'from manim import *', with no Markdown fences."
    ),
    "question_to_manim_code": (
        "Solve and visually teach the following calculus problem in one complete "
        "executable Manim Community Edition Python scene. Return Python code only, "
        "beginning with 'from manim import *', with no Markdown fences."
    ),
    "question_solution_to_manim_code": (
        "Use the supplied calculus question and correct solution to create one complete "
        "executable Manim Community Edition Python scene. Return Python code only, "
        "beginning with 'from manim import *', with no Markdown fences."
    ),
    "storyboard_critique": (
        "Critique the following calculus animation storyboard for mathematical and "
        "visual correctness. Return only the requested structured critique."
    ),
}

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

SHAREGPT_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT of a Hugging Face Qwen2.5-Coder checkpoint."
    )

    hub = parser.add_argument_group("Hugging Face inputs and outputs")
    hub.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    hub.add_argument("--model-revision", default=None)
    hub.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    hub.add_argument(
        "--local-dataset-file",
        type=Path,
        default=None,
        help=(
            "Read a local JSONL file instead of loading a Hugging Face dataset. "
            "This is convenient when the dataset is uploaded to Google Colab."
        ),
    )
    hub.add_argument(
        "--dataset-file",
        default=None,
        help=(
            "Load one Hub JSONL file directly. The calculus default automatically "
            f"uses {DEFAULT_DATASET_FILE!r} to remain compatible with Unsloth's "
            "datasets version."
        ),
    )
    hub.add_argument("--dataset-config", default=None)
    hub.add_argument("--dataset-revision", default=None)
    hub.add_argument("--train-split", default="train")
    hub.add_argument(
        "--eval-split",
        default=None,
        help="Existing validation split. If omitted, --validation-size is used.",
    )
    hub.add_argument(
        "--hub-model-id",
        default=None,
        help="Optional destination repo, for example USER/qwen-calculus-sft.",
    )
    hub.add_argument(
        "--hub-save-method",
        choices=("lora", "merged_16bit"),
        default="merged_16bit",
        help="Artifact to push when --hub-model-id is set.",
    )
    hub.add_argument("--hub-private", action="store_true")

    data = parser.add_argument_group("Dataset")
    data.add_argument(
        "--validation-size",
        type=float,
        default=0.03,
        help="Seeded validation fraction when --eval-split is not supplied.",
    )
    data.add_argument("--max-seq-length", type=int, default=8192)
    data.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Allow rows longer than --max-seq-length to be truncated.",
    )
    data.add_argument("--dataset-num-proc", type=int, default=2)
    data.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional shuffled subset for a smoke test.",
    )

    lora = parser.add_argument_group("QLoRA")
    lora.add_argument("--lora-rank", type=int, default=32)
    lora.add_argument("--lora-alpha", type=int, default=32)
    lora.add_argument("--lora-dropout", type=float, default=0.0)
    lora.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    training = parser.add_argument_group("Training")
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--eval-batch-size", type=int, default=2)
    training.add_argument("--gradient-accumulation-steps", type=int, default=8)
    training.add_argument("--epochs", type=float, default=2.0)
    training.add_argument("--learning-rate", type=float, default=5e-5)
    training.add_argument("--warmup-ratio", type=float, default=0.05)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--logging-steps", type=int, default=10)
    training.add_argument(
        "--checkpoint-steps",
        type=int,
        default=250,
        help="Evaluate and save every N optimizer steps.",
    )
    training.add_argument(
        "--checkpoint-strategy",
        choices=("steps", "epoch"),
        default="steps",
        help=(
            "Evaluate/save by optimizer steps (the default) or once per epoch. "
            "Epoch checkpoints are useful for the 6,023-row report reproduction."
        ),
    )
    training.add_argument("--save-total-limit", type=int, default=2)
    training.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pack short rows together; disabled by default for broad GPU support.",
    )
    training.add_argument("--report-to", default="none")
    training.add_argument(
        "--live-table",
        action="store_true",
        help="Show an in-place HTML training dashboard in Jupyter or Colab.",
    )
    training.add_argument("--seed", type=int, default=SEED)
    training.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="Resume from PATH, or from the latest checkpoint when no PATH is given.",
    )
    training.add_argument(
        "--save-merged-model",
        action="store_true",
        help="Also write output-dir/merged_16bit after training.",
    )

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.local_dataset_file is not None and args.dataset_file is not None:
        parser.error("--local-dataset-file and --dataset-file are mutually exclusive")
    positive = {
        "--max-seq-length": args.max_seq_length,
        "--dataset-num-proc": args.dataset_num_proc,
        "--lora-rank": args.lora_rank,
        "--lora-alpha": args.lora_alpha,
        "--batch-size": args.batch_size,
        "--eval-batch-size": args.eval_batch_size,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--epochs": args.epochs,
        "--learning-rate": args.learning_rate,
        "--logging-steps": args.logging_steps,
        "--checkpoint-steps": args.checkpoint_steps,
        "--save-total-limit": args.save_total_limit,
    }
    for flag, value in positive.items():
        if value <= 0:
            parser.error(f"{flag} must be greater than zero")
    if not 0 <= args.validation_size < 1:
        parser.error("--validation-size must be in [0, 1)")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora-dropout must be in [0, 1)")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        parser.error("--max-train-samples must be greater than zero")


def _normalise_message(message: dict[str, Any], row_index: int) -> dict[str, str]:
    """Convert either OpenAI/Qwen or ShareGPT message keys to one schema."""
    if not isinstance(message, dict):
        raise ValueError(f"row {row_index}: every message must be an object")
    raw_role = message.get("role", message.get("from"))
    content = message.get("content", message.get("value"))
    role = SHAREGPT_ROLES.get(raw_role)

    if role is None:
        raise ValueError(f"row {row_index}: unsupported role {raw_role!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"row {row_index}: every message needs non-empty text")
    return {"role": role, "content": content}


def _normalise_text(value: Any, row_index: int, field: str) -> str:
    if isinstance(value, str):
        text = value
    elif value is not None:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        raise ValueError(f"row {row_index}: missing calculus field {field!r}")
    if not text.strip():
        raise ValueError(f"row {row_index}: calculus field {field!r} is empty")
    return text


def _calculus_training_pair(
    example: dict[str, Any], row_index: int, group_question: str | None = None
) -> tuple[str, str, str, str]:
    """Extract one explicit task, input, and target from a calculus record."""
    raw_task = example.get("task_type")
    task = (
        "question_solution_to_manim_code"
        if raw_task == "question_solution_to_manim"
        else raw_task
    )
    if task not in CALCULUS_TASK_INSTRUCTIONS:
        raise ValueError(f"row {row_index}: unsupported calculus task {raw_task!r}")

    if "input" in example and "output" in example:
        source = _normalise_text(example["input"], row_index, "input")
        target = _normalise_text(example["output"], row_index, "output")
    elif task == "question_to_solution":
        source = _normalise_text(example.get("question"), row_index, "question")
        target = _normalise_text(example.get("solution"), row_index, "solution")
    elif task == "question_to_storyboard":
        source = _normalise_text(example.get("question"), row_index, "question")
        target = _normalise_text(example.get("storyboard"), row_index, "storyboard")
    elif task == "solution_to_storyboard":
        solution = _normalise_text(example.get("solution"), row_index, "solution")
        question = example.get("question") or group_question
        if question:
            source = f"Problem:\n{question}\n\nWorked solution:\n{solution}"
        else:
            source = solution
        target = _normalise_text(example.get("storyboard"), row_index, "storyboard")
    elif task == "storyboard_to_manim_code":
        source = _normalise_text(example.get("storyboard"), row_index, "storyboard")
        target = _normalise_text(example.get("manim_code"), row_index, "manim_code")
    elif task == "question_to_manim_code":
        source = _normalise_text(example.get("question"), row_index, "question")
        target = _normalise_text(example.get("manim_code"), row_index, "manim_code")
    elif task == "question_solution_to_manim_code":
        question = _normalise_text(example.get("question"), row_index, "question")
        solution = _normalise_text(example.get("solution"), row_index, "solution")
        source = f"Question:\n{question}\n\nReference solution:\n{solution}"
        target = _normalise_text(example.get("manim_code"), row_index, "manim_code")
    else:
        raise ValueError(
            f"row {row_index}: task {raw_task!r} needs explicit input/output fields"
        )

    instruction = example.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = CALCULUS_TASK_INSTRUCTIONS[task]
    if task == "solution_to_storyboard" and group_question and not source.startswith(
        "Problem:\n"
    ):
        source = f"Problem:\n{group_question}\n\nWorked solution:\n{source}"
    return task, instruction.strip(), source, target


def _calculus_group_id(example: dict[str, Any], row_index: int) -> str:
    """Group aligned task variants so validation cannot see a training problem."""
    metadata = example.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    canonical_id = example.get("canonical_problem_id") or metadata.get(
        "canonical_problem_id"
    )
    if canonical_id:
        return f"canonical:{canonical_id}"

    record_id = str(example.get("id", f"row-{row_index}"))
    match = re.search(r"(\d+)$", record_id)
    topic = str(example.get("topic", "unknown"))
    if match and topic != "Functions and prerequisites":
        return f"{topic}:{match.group(1)}"

    source_id = str(metadata.get("combined_source_record_id", ""))
    source_match = re.search(r"(\d+)$", source_id)
    legacy_prefixes = ("calc_sol_", "calc_qsb_", "calc_ssb_", "calc_sbc_", "calc_v2_")
    if source_match and source_id.startswith(legacy_prefixes):
        return f"functions-legacy:{source_match.group(1)}"
    return f"record:{record_id}"


def to_prompt_completion(
    example: dict[str, Any],
    row_index: int,
    question_lookup: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make the last assistant turn the only supervised completion."""
    raw_messages = example.get("messages", example.get("conversations"))
    if raw_messages is None and "task_type" in example:
        group_id = _calculus_group_id(example, row_index)
        group_question = (question_lookup or {}).get(group_id)
        _, instruction, source, target = _calculus_training_pair(
            example,
            row_index,
            group_question=group_question,
        )
        return {
            "prompt": [
                {"role": "system", "content": CALCULUS_SYSTEM_PROMPT},
                {"role": "user", "content": f"{instruction}\n\nInput:\n{source}"},
            ],
            "completion": [{"role": "assistant", "content": target}],
            "_split_group": group_id,
        }

    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        raise ValueError(
            f"row {row_index}: expected a messages or conversations list"
        )

    messages = [_normalise_message(message, row_index) for message in raw_messages]
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"row {row_index}: the final message must be assistant")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise ValueError(f"row {row_index}: the prompt needs at least one user turn")

    return {
        "prompt": messages[:-1],
        "completion": messages[-1:],
        "_split_group": f"record:{example.get('id', row_index)}",
    }


def _calculus_question_lookup(dataset: Any) -> dict[str, str]:
    """Recover problem statements needed by solution-to-storyboard records."""
    ranked: dict[str, tuple[int, str]] = {}
    priorities = {
        "question_to_solution": 0,
        "question_to_storyboard": 1,
        "question_to_manim_code": 2,
    }
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None and "task_type" not in column_names:
        return {}
    for index, example in enumerate(dataset):
        task = example.get("task_type")
        if task not in priorities:
            continue
        question = example.get("question", example.get("input"))
        if not isinstance(question, str) or not question.strip():
            continue
        group_id = _calculus_group_id(example, index)
        candidate = (priorities[task], question.strip())
        if group_id not in ranked or candidate[0] < ranked[group_id][0]:
            ranked[group_id] = candidate
    return {group_id: value for group_id, (_, value) in ranked.items()}


def convert_dataset(dataset: Any, num_proc: int, description: str) -> Any:
    supported = {"messages", "conversations", "task_type"}
    if not (supported & set(dataset.column_names)):
        raise ValueError(
            "Dataset must contain either a 'messages' column (role/content) or "
            "a 'conversations' column (from/value), or calculus task records."
        )
    question_lookup = _calculus_question_lookup(dataset)
    return dataset.map(
        to_prompt_completion,
        with_indices=True,
        fn_kwargs={"question_lookup": question_lookup},
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=description,
    )


def grouped_train_test_split(
    dataset: Any, validation_size: float, seed: int
) -> tuple[Any, Any]:
    """Create a deterministic split without separating aligned calculus tasks."""
    group_to_indices: dict[str, list[int]] = {}
    for index, group_id in enumerate(dataset["_split_group"]):
        group_to_indices.setdefault(group_id, []).append(index)
    if len(group_to_indices) < 2:
        raise ValueError("At least two independent problem groups are needed to split.")

    ordered_groups = sorted(
        group_to_indices,
        key=lambda group_id: hashlib.sha256(
            f"{seed}\0{group_id}".encode("utf-8")
        ).hexdigest(),
    )
    target_rows = max(1, round(len(dataset) * validation_size))
    evaluation_groups: set[str] = set()
    evaluation_rows = 0
    for group_id in ordered_groups:
        if evaluation_rows >= target_rows:
            break
        evaluation_groups.add(group_id)
        evaluation_rows += len(group_to_indices[group_id])

    train_indices = []
    evaluation_indices = []
    for group_id, indices in group_to_indices.items():
        destination = evaluation_indices if group_id in evaluation_groups else train_indices
        destination.extend(indices)
    if not train_indices or not evaluation_indices:
        raise ValueError("The grouped validation split left an empty partition.")

    print(
        f"Grouped split: {len(group_to_indices) - len(evaluation_groups):,} train "
        f"problem groups and {len(evaluation_groups):,} validation problem groups."
    )
    train = dataset.select(sorted(train_indices)).remove_columns("_split_group")
    evaluation = dataset.select(sorted(evaluation_indices)).remove_columns(
        "_split_group"
    )
    return train, evaluation


def load_and_prepare_datasets(
    args: argparse.Namespace,
    load_dataset: Any,
    dataset_from_list: Any,
    hf_hub_download: Any,
) -> tuple[Any, Any]:
    dataset_file = args.dataset_file
    if (
        args.local_dataset_file is None
        and dataset_file is None
        and args.dataset_id == DEFAULT_DATASET_ID
    ):
        dataset_file = DEFAULT_DATASET_FILE

    if args.local_dataset_file is not None or dataset_file:
        if args.eval_split:
            raise ValueError(
                "--eval-split cannot be combined with a JSONL file; use the grouped "
                "--validation-size split instead."
            )
        if args.train_split != "train":
            raise ValueError("JSONL file loading supports only --train-split train")

        if args.local_dataset_file is not None:
            local_path = args.local_dataset_file.expanduser().resolve()
            if not local_path.is_file():
                raise FileNotFoundError(f"Local dataset file not found: {local_path}")
            dataset_source = str(local_path)
        else:
            local_path = Path(
                hf_hub_download(
                    repo_id=args.dataset_id,
                    filename=dataset_file,
                    repo_type="dataset",
                    revision=args.dataset_revision,
                )
            )
            dataset_source = f"{args.dataset_id}/{dataset_file}"
        raw_rows = []
        with local_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {dataset_source} at line {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected a JSON object in {dataset_source} at line {line_number}"
                    )
                raw_rows.append(row)
        if not raw_rows:
            raise ValueError(f"The dataset file {dataset_source!r} is empty")

        print(f"Loaded {len(raw_rows):,} raw rows directly from {dataset_source}.")
        question_lookup = _calculus_question_lookup(raw_rows)
        formatted_rows = [
            to_prompt_completion(row, index, question_lookup)
            for index, row in enumerate(raw_rows)
        ]
        train = dataset_from_list(formatted_rows)
    else:
        load_kwargs = {"revision": args.dataset_revision}
        load_kwargs = {
            key: value for key, value in load_kwargs.items() if value is not None
        }

        train = load_dataset(
            args.dataset_id,
            args.dataset_config,
            split=args.train_split,
            **load_kwargs,
        )
        train = convert_dataset(
            train, args.dataset_num_proc, "Formatting training chats"
        )

    if args.max_train_samples is not None and args.max_train_samples < len(train):
        train = train.shuffle(seed=args.seed).select(range(args.max_train_samples))
    if args.eval_split:
        load_kwargs = {"revision": args.dataset_revision}
        load_kwargs = {
            key: value for key, value in load_kwargs.items() if value is not None
        }
        evaluation = load_dataset(
            args.dataset_id,
            args.dataset_config,
            split=args.eval_split,
            **load_kwargs,
        )
        evaluation = convert_dataset(
            evaluation,
            args.dataset_num_proc,
            "Formatting validation chats",
        )
        train = train.remove_columns("_split_group")
        evaluation = evaluation.remove_columns("_split_group")
    elif args.validation_size:
        train, evaluation = grouped_train_test_split(
            train,
            validation_size=args.validation_size,
            seed=args.seed,
        )
    else:
        evaluation = None
        train = train.remove_columns("_split_group")
    return train, evaluation


def validate_sequence_lengths(
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
    allow_truncation: bool,
) -> None:
    datasets = [train_dataset]
    if eval_dataset is not None:
        datasets.append(eval_dataset)
    if not len(train_dataset):
        raise ValueError("The selected training split is empty.")

    lengths = []
    for dataset in datasets:
        for row in dataset:
            messages = row["prompt"] + row["completion"]
            lengths.append(
                len(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                )
            )

    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        index = max(0, int(len(ordered) * fraction + 0.999999) - 1)
        return ordered[index]

    too_long = sum(length > max_seq_length for length in ordered)
    print(
        f"Prepared {len(train_dataset):,} train and "
        f"{0 if eval_dataset is None else len(eval_dataset):,} validation rows. "
        f"Tokens: p50={percentile(0.50):,}, p95={percentile(0.95):,}, "
        f"max={ordered[-1]:,}; above limit={too_long:,}."
    )
    if too_long and not allow_truncation:
        raise ValueError(
            f"{too_long:,} rows exceed max_seq_length={max_seq_length:,}. "
            "Increase --max-seq-length or explicitly pass --allow-truncation."
        )


def save_or_push_model(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    final_adapter_dir: Path,
) -> None:
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"Saved LoRA adapter to {final_adapter_dir}")

    if args.save_merged_model:
        merged_dir = args.output_dir / "merged_16bit"
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Saved merged 16-bit model to {merged_dir}")

    if not args.hub_model_id:
        return
    if args.hub_save_method == "merged_16bit":
        model.push_to_hub_merged(
            args.hub_model_id,
            tokenizer,
            save_method="merged_16bit",
            private=args.hub_private,
        )
    else:
        model.push_to_hub(args.hub_model_id, private=args.hub_private)
        tokenizer.push_to_hub(args.hub_model_id, private=args.hub_private)
    print(f"Pushed {args.hub_save_method} artifact to {args.hub_model_id}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_report_artifacts(
    args: argparse.Namespace,
    trainer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any] | None,
) -> None:
    """Save compact, report-ready evidence alongside the trainer checkpoints."""
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = [_json_safe(row) for row in trainer.state.log_history]
    history_path = args.output_dir / "training_history.csv"
    preferred_columns = [
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "train_runtime",
        "eval_runtime",
    ]
    discovered = sorted({key for row in history for key in row})
    columns = preferred_columns + [
        column for column in discovered if column not in preferred_columns
    ]
    with history_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )

    dataset_details: dict[str, Any]
    if args.local_dataset_file is not None:
        dataset_path = args.local_dataset_file.expanduser().resolve()
        dataset_details = {
            "source": "local_jsonl",
            "path": str(dataset_path),
            "filename": dataset_path.name,
            "bytes": dataset_path.stat().st_size,
            "sha256": _file_sha256(dataset_path),
        }
    elif args.dataset_file:
        dataset_details = {
            "source": "huggingface_jsonl",
            "dataset_id": args.dataset_id,
            "filename": args.dataset_file,
            "revision": args.dataset_revision,
        }
    else:
        dataset_details = {
            "source": "huggingface_dataset",
            "dataset_id": args.dataset_id,
            "config": args.dataset_config,
            "revision": args.dataset_revision,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
        }

    gpu_details = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        gpu_details = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        }

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "method": "QLoRA SFT with rsLoRA",
            "final_adapter": str((args.output_dir / "final_adapter").resolve()),
        },
        "dataset": {
            **dataset_details,
            "train_rows": len(train_dataset),
            "validation_rows": 0 if eval_dataset is None else len(eval_dataset),
            "validation_fraction_requested": args.validation_size,
        },
        "configuration": _json_safe(vars(args)),
        "derived_configuration": {
            "effective_batch_size_one_gpu": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "completion_only_loss": True,
            "loss_type": "nll",
            "lora_target_modules": list(LORA_TARGET_MODULES),
            "seed": args.seed,
        },
        "result": {
            "global_step": trainer.state.global_step,
            "completed_epoch": trainer.state.epoch,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "train_metrics": _json_safe(train_metrics),
            "eval_metrics": _json_safe(eval_metrics),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "unsloth",
                    "unsloth_zoo",
                    "torch",
                    "transformers",
                    "trl",
                    "datasets",
                    "peft",
                    "accelerate",
                )
            },
            "cuda_version": torch.version.cuda,
            "gpu": gpu_details,
        },
        "files": {
            "training_history_csv": str(history_path.resolve()),
            "trainer_state_json": str((args.output_dir / "trainer_state.json").resolve()),
            "train_results_json": str((args.output_dir / "train_results.json").resolve()),
            "eval_results_json": str((args.output_dir / "eval_results.json").resolve()),
        },
        "interpretation_warning": (
            "Training and validation loss document optimization behavior; they do not "
            "by themselves establish Manim render success, mathematical correctness, "
            "or improvement on held-out prompts."
        ),
    }
    summary_path = args.output_dir / "sft_report_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved report-ready history to {history_path}")
    print(f"Saved report summary to {summary_path}")


def make_live_training_callback() -> Any:
    """Build a lazy Colab/Jupyter callback without importing Transformers early."""
    import time

    import torch
    from IPython.display import HTML, display
    from transformers import TrainerCallback

    class LiveTrainingCallback(TrainerCallback):
        def __init__(self) -> None:
            self.display_handle = None
            self.last_rendered_at = 0.0
            self.train_loss: float | None = None
            self.eval_loss: float | None = None
            self.learning_rate: float | None = None
            self.status = "Starting"

        @staticmethod
        def _number(value: float | None, precision: int = 4) -> str:
            return "—" if value is None else f"{value:.{precision}f}"

        def _render(self, state: Any, *, force: bool = False) -> None:
            if not state.is_world_process_zero:
                return
            now = time.monotonic()
            if not force and now - self.last_rendered_at < 1.0:
                return
            self.last_rendered_at = now

            step = int(state.global_step)
            max_steps = max(int(state.max_steps), 1)
            progress = min(100.0, 100.0 * step / max_steps)
            epoch = float(state.epoch or 0.0)
            gpu_memory = (
                torch.cuda.memory_reserved() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            dashboard = HTML(
                f"""
                <div style="font-family:system-ui;max-width:1050px">
                  <h3 style="margin:0 0 10px">Live SFT training</h3>
                  <div style="height:8px;background:#e5e7eb;border-radius:6px;overflow:hidden">
                    <div style="width:{progress:.2f}%;height:100%;background:#2563eb"></div>
                  </div>
                  <div style="margin:5px 0 12px;color:#4b5563">
                    {progress:.1f}% complete · progress refreshes continuously; metrics refresh at logging intervals
                  </div>
                  <table style="border-collapse:collapse;width:100%;text-align:right">
                    <thead><tr style="background:#f3f4f6">
                      <th style="padding:8px;text-align:left">Status</th>
                      <th style="padding:8px">Step</th>
                      <th style="padding:8px">Epoch</th>
                      <th style="padding:8px">Train loss</th>
                      <th style="padding:8px">Validation loss</th>
                      <th style="padding:8px">Learning rate</th>
                      <th style="padding:8px">GPU reserved</th>
                    </tr></thead>
                    <tbody><tr style="border-bottom:1px solid #e5e7eb">
                      <td style="padding:8px;text-align:left">{self.status}</td>
                      <td style="padding:8px">{step:,} / {max_steps:,}</td>
                      <td style="padding:8px">{epoch:.3f}</td>
                      <td style="padding:8px">{self._number(self.train_loss)}</td>
                      <td style="padding:8px">{self._number(self.eval_loss)}</td>
                      <td style="padding:8px">{self._number(self.learning_rate, 7)}</td>
                      <td style="padding:8px">{gpu_memory:.2f} GiB</td>
                    </tr></tbody>
                  </table>
                </div>
                """
            )
            if self.display_handle is None:
                self.display_handle = display(dashboard, display_id=True)
            else:
                self.display_handle.update(dashboard)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.status = "Training"
            self._render(state, force=True)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.status = (
                "Complete" if state.global_step >= state.max_steps else "Training"
            )
            self._render(state)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, float] | None = None,
            **kwargs: Any,
        ) -> None:
            logs = logs or {}
            if "loss" in logs:
                self.train_loss = float(logs["loss"])
            if "eval_loss" in logs:
                self.eval_loss = float(logs["eval_loss"])
            if "learning_rate" in logs:
                self.learning_rate = float(logs["learning_rate"])
            self._render(state, force=True)

        def on_prediction_step(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            self.status = "Validating"
            self._render(state)

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, float] | None = None,
            **kwargs: Any,
        ) -> None:
            metrics = metrics or {}
            if "eval_loss" in metrics:
                self.eval_loss = float(metrics["eval_loss"])
            self.status = (
                "Complete" if state.global_step >= state.max_steps else "Training"
            )
            self._render(state, force=True)

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.status = "Checkpoint saved"
            self._render(state, force=True)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.status = "Complete"
            self._render(state, force=True)

    return LiveTrainingCallback()


def main() -> None:
    args = parse_args()

    # Unsloth must be imported before Transformers, TRL, or PEFT so its patches
    # are installed before those libraries initialise model classes.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset, load_dataset
    from huggingface_hub import hf_hub_download
    from trl import SFTConfig, SFTTrainer

    # Validate and split the inexpensive dataset before allocating GPU model memory.
    train_dataset, eval_dataset = load_and_prepare_datasets(
        args,
        load_dataset,
        Dataset.from_list,
        hf_hub_download,
    )

    model_kwargs = {}
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        **model_kwargs,
    )
    FastLanguageModel.for_training(model)

    if getattr(model, "peft_config", None):
        print("Loaded an existing PEFT adapter; continuing that adapter.")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(LORA_TARGET_MODULES),
            use_gradient_checkpointing="unsloth",
            use_rslora=True,
            random_state=args.seed,
        )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if not trainable_parameters:
        raise RuntimeError("The loaded model has no trainable adapter parameters.")
    print(f"Trainable adapter parameters: {trainable_parameters:,}")

    validate_sequence_lengths(
        train_dataset,
        eval_dataset,
        tokenizer,
        args.max_seq_length,
        args.allow_truncation,
    )

    has_eval = eval_dataset is not None
    periodic_strategy = args.checkpoint_strategy
    training_config = SFTConfig(
        output_dir=str(args.output_dir),
        run_name=args.output_dir.name,
        max_length=args.max_seq_length,
        eos_token=tokenizer.eos_token,
        completion_only_loss=True,
        loss_type="nll",
        packing=args.packing,
        packing_strategy="bfd",
        eval_packing=False,
        dataset_num_proc=args.dataset_num_proc,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        gradient_checkpointing=True,
        eval_strategy=periodic_strategy if has_eval else "no",
        eval_steps=(
            args.checkpoint_steps
            if has_eval and periodic_strategy == "steps"
            else None
        ),
        save_strategy=periodic_strategy,
        save_steps=args.checkpoint_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        disable_tqdm=args.live_table,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )
    callbacks = [make_live_training_callback()] if args.live_table else None
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_config,
        callbacks=callbacks,
    )
    if args.live_table:
        # Avoid duplicate dictionary logs beneath the in-place notebook dashboard.
        from transformers import PrinterCallback

        trainer.remove_callback(PrinterCallback)

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    if trainer.state.best_model_checkpoint:
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    evaluation_metrics = None
    if has_eval:
        evaluation_metrics = trainer.evaluate()
        trainer.save_metrics("eval", evaluation_metrics)
        # Preserve the final selected-checkpoint evaluation in trainer_state.json.
        trainer.save_state()
    save_report_artifacts(
        args,
        trainer,
        train_dataset,
        eval_dataset,
        result.metrics,
        evaluation_metrics,
    )
    save_or_push_model(
        args,
        trainer.model,
        tokenizer,
        args.output_dir / "final_adapter",
    )


if __name__ == "__main__":
    main()
