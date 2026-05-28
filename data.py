# data.py
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset, concatenate_datasets
import pandas as pd

# for language modeling problems how long to use the prefix as
PREFIX_LENGTH: int = 100
SPECBENCH_DEFAULT_PATH = "data/spec_bench/question.jsonl"
GOVREPORT_DEFAULT_PATH = "data/govreport/govreport_16K.jsonl"
PG19_DEFAULT_PATH = "data/pg19/pg19_16K.jsonl"
BOOKSUM_DEFAULT_PATH = "data/booksum/booksum_16K.jsonl"

@dataclass
class EvaluationExample:
    input: str
    output: str
    metadata: Optional[Dict] = None


class DatasetFormat:
    AIME24: str = "aime24"
    AIME25: str = "aime25"
    GOVREPORT: str = "govreport"
    PG19: str = "pg19"
    BOOKSUM: str = "booksum"
    GPQA: str = "gpqa"
    MMLU_PRO: str = "mmlu_pro"


AIME24_DEFAULT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only the answer) inside \\boxed{{}}.\n\n{message}\n"
    " <think> Let's think step by step. </think>\n"
)
AIME25_DEFAULT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only the answer) inside \\boxed{{}}.\n\n{message}\n"
    " <think> Let's think step by step. </think>\n"
)


def get_valid_dataset_formats():
    """Get all available dataset formats."""
    return [value for key, value in DatasetFormat.__dict__.items() if not key.startswith('__')]


def apply_template(message: str, template: str = None) -> str:
    """
    Applies a template to a given message.
    
    Parameters:
        message (str): The message to insert into the template.
        template (str): The template with a placeholder for the message in `{message}`.
        
    Returns:
        str: The formatted message with the template applied.
    """
    if template is None:
        return message
    return template.format(message=message)


def prepare_aime24_format(template: str = None) -> List[EvaluationExample]:
    """
    Prepare AIME 2024 dataset (HuggingFaceH4/aime_2024).
    """
    evaluation_data_points = []
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    if template is None:
        template = AIME24_DEFAULT_TEMPLATE

    for i, dp in enumerate(ds):
        problem = dp["problem"]
        answer = str(int(dp["answer"]))
        prompt = apply_template(message=problem, template=template)
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=answer,
                metadata={"id": i, "type": "math_reasoning_aime24"}
            )
        )
    return evaluation_data_points


def prepare_aime25_format(template: str = None) -> List[EvaluationExample]:
    """
    Prepare AIME 2025 dataset ("opencompass/AIME2025").
    Loads both AIME I and AIME II.
    """
    evaluation_data_points = []

    try:
        ds_1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds_2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        ds = concatenate_datasets([ds_1, ds_2])
    except ValueError:
        ds_1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="train")
        ds_2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="train")
        ds = concatenate_datasets([ds_1, ds_2])

    if template is None:
        template = AIME25_DEFAULT_TEMPLATE

    for i, dp in enumerate(ds):
        problem = dp["question"]
        raw_answer = dp.get("answer", "")
        answer = str(raw_answer).strip()
        
        try:
            if str(answer).isdigit():
                answer = str(int(answer))
        except:
            pass

        prompt = apply_template(message=problem, template=template)
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=answer,
                metadata={"id": i, "type": "math_reasoning_aime25"}
            )
        )
    return evaluation_data_points


def prepare_govreport_format(
    data_path: str = GOVREPORT_DEFAULT_PATH,
    template: str = None,
) -> List[EvaluationExample]:
    evaluation_data_points = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            prompt_text = row.get("text", "")
            # Add instruction for summarization
            prompt_text = f"Summarize the following text:\n\n{prompt_text}\n\nSummary:"
            answer_text = row.get("answer", "")

            prompt = apply_template(message=prompt_text, template=template)

            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=answer_text if str(answer_text).startswith(" ") else f" {answer_text}",
                    metadata={
                        "id": row.get("id", i),
                        "type": "govreport",
                        "original_length": row.get("original_length", None),
                        "trunc_length": row.get("trunc_length", None),
                    },
                )
            )
    return evaluation_data_points


def prepare_pg19_format(
    data_path: str = PG19_DEFAULT_PATH,
    template: str = None,
) -> List[EvaluationExample]:
    evaluation_data_points = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            prompt_text = row.get("text", "")
            # Add instruction for summarization
            prompt_text = f"Summarize the following text:\n\n{prompt_text}\n\nSummary:"
            answer_text = row.get("answer", "")

            prompt = apply_template(message=prompt_text, template=template)

            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=answer_text if str(answer_text).startswith(" ") else f" {answer_text}",
                    metadata={
                        "id": row.get("id", i),
                        "type": "pg19",
                        "original_length": row.get("original_length", None),
                        "trunc_length": row.get("trunc_length", None),
                    },
                )
            )
    return evaluation_data_points


def prepare_booksum_format(
    data_path: str = BOOKSUM_DEFAULT_PATH,
    template: str = None,
) -> List[EvaluationExample]:
    evaluation_data_points = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            prompt_text = row.get("text", "")
            # Add instruction for summarization
            prompt_text = f"Summarize the following text:\n\n{prompt_text}\n\nSummary:"
            answer_text = row.get("answer", "")

            prompt = apply_template(message=prompt_text, template=template)

            evaluation_data_points.append(
                EvaluationExample(
                    input=prompt,
                    output=answer_text if str(answer_text).startswith(" ") else f" {answer_text}",
                    metadata={
                        "id": row.get("id", i),
                        "type": "booksum",
                        "original_length": row.get("original_length", None),
                        "trunc_length": row.get("trunc_length", None),
                    },
                )
            )
    return evaluation_data_points


def prepare_gpqa_format(template: str = None) -> List[EvaluationExample]:
    """Prepare GPQA dataset."""
    evaluation_data_points = []
    # Using 'train' split as per user reference
    try:
        dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond', split='train')
    except Exception as e:
        print(f"Warning: Could not load GPQA dataset: {e}")
        return []

    for i, row in enumerate(dataset):
        choices = [
            row['Correct Answer'],
            row['Incorrect Answer 1'], 
            row['Incorrect Answer 2'], 
            row['Incorrect Answer 3'], 
        ]
        indexed_choices = list(enumerate(choices))
        random.shuffle(indexed_choices)

        labels = ['A', 'B', 'C', 'D']
        shuffled_text = ''
        correct_label = ''
        for j, (original_idx, text) in enumerate(indexed_choices):
            shuffled_text += f"{labels[j]}) {text}\n"
            if original_idx == 0:
                correct_label = labels[j]

        prompt_content = f"""Answer the following multiple-choice question. Think step-by-step before providing your final answer.
    
        Question: {row['Question']}

        Options:
        {shuffled_text}
        The last line of your response MUST be in the following format: "Answer: [LETTER]" where [LETTER] is one of {{A, B, C, D}}."""
        
        prompt = apply_template(message=prompt_content, template=template)
        
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=correct_label,
                metadata={"id": i, "type": "gpqa"}
            )
        )
    return evaluation_data_points


def prepare_mmlu_pro_format(template: str = None) -> List[EvaluationExample]:
    """
    Prepare MMLU-Pro dataset ("TIGER-Lab/MMLU-Pro").
    Ref: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    """
    evaluation_data_points = []
    try:
        # Load validation set (smaller than test set)
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    except Exception as e:
        print(f"Warning: Could not load MMLU-Pro dataset: {e}")
        return []

    for i, row in enumerate(dataset):
        # Fields: question, options, answer, answer_index, cot_content, category
        question = row["question"]
        options = row["options"]
        raw_answer = row["answer"]  # e.g. "C"
        
        # Format options as A, B, C...
        option_text = ""
        labels = []
        for idx, opt in enumerate(options):
            label = chr(65 + idx) # A, B, C...
            labels.append(label)
            option_text += f"{label}. {opt}\n"
        
        prompt_content = (
            f"Question:\n{question}\n\n"
            f"Options:\n{option_text}\n"
            "Answer the question by selecting the correct option. "
            "Think step by step and then provide your final answer in the format: \"Answer: [OPTION]\"."
        )
        
        prompt = apply_template(message=prompt_content, template=template)
        
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=raw_answer,
                metadata={
                    "id": row.get("question_id", i),
                    "type": "mmlu_pro",
                    "category": row.get("category", "unknown") 
                }
            )
        )

    return evaluation_data_points


def get_dataset(
    dataset_format: str,
    num_samples: int = None,
    random_shuffle: bool = True,
    seed: int = 42,
    data_path: Optional[str] = None,
    n_shot: int = 0,
    prompt_field: str = "prompt",
    response_field: str = "response",
    template: str = None
) -> List[EvaluationExample]:
    """
    Get dataset based on format specification.
    
    Args:
        dataset_format: One of the DatasetFormat values
        num_samples: Number of samples to return (None for all)
        random_shuffle: Whether to shuffle the dataset
        seed: Random seed for reproducibility
        data_path: Path to custom dataset file (for CUSTOM_JSONL)
        n_shot: Number of few-shot examples (for applicable datasets)
        prompt_field: Field name for prompts in custom dataset
        response_field: Field name for responses in custom dataset
        template: Template to apply to prompts
        
    Returns:
        List of EvaluationExample objects
    """
    random.seed(seed)
    
    if dataset_format == DatasetFormat.AIME24:
        evaluation_data_points = prepare_aime24_format(template=template)
    elif dataset_format == DatasetFormat.AIME25:
        evaluation_data_points = prepare_aime25_format(template=template)
    elif dataset_format == DatasetFormat.CODEFORCES:
        evaluation_data_points = prepare_codeforces_format(template=template)
    elif dataset_format == DatasetFormat.CUSTOM_JSONL:
        if data_path is None:
            raise ValueError("data_path is required for CUSTOM_JSONL format")
        evaluation_data_points = prepare_custom_jsonl_format(
            data_path, prompt_field=prompt_field, response_field=response_field, template=template
        )
    elif dataset_format == DatasetFormat.SPEC_BENCH:
        evaluation_data_points = prepare_spec_bench_format(
            data_path=SPECBENCH_DEFAULT_PATH,
            template=template,
        )
    elif dataset_format == DatasetFormat.GOVREPORT:
        evaluation_data_points = prepare_govreport_format(
            data_path=data_path if data_path is not None else GOVREPORT_DEFAULT_PATH,
            template=template,
        )
    elif dataset_format == DatasetFormat.PG19:
        evaluation_data_points = prepare_pg19_format(
            data_path=data_path if data_path is not None else PG19_DEFAULT_PATH,
            template=template,
        )
    elif dataset_format == DatasetFormat.BOOKSUM:
        evaluation_data_points = prepare_booksum_format(
            data_path=data_path if data_path is not None else BOOKSUM_DEFAULT_PATH,
            template=template,
        )
    elif dataset_format == DatasetFormat.GPQA:
        evaluation_data_points = prepare_gpqa_format(template=template)
    elif dataset_format == DatasetFormat.MMLU_PRO:
        evaluation_data_points = prepare_mmlu_pro_format(template=template)
    else:
        raise NotImplementedError(f"Unknown dataset format: {dataset_format}")

    if random_shuffle:
        random.shuffle(evaluation_data_points)

    if num_samples is not None and num_samples > 0:
        evaluation_data_points = evaluation_data_points[:num_samples]

    return evaluation_data_points


def get_dataset_info(dataset_format: str) -> Dict:
    """Get information about a dataset format."""
    info = {
        DatasetFormat.AIME24: {
            "description": "AIME 2024 math reasoning (integer answers in \\boxed{})",
            "task_type": "math_reasoning",
            "supports_few_shot": False,
            "typical_input_length": "50-300 tokens",
            "typical_output_length": "1-20 tokens"
        },
        DatasetFormat.AIME25: {
            "description": "AIME 2025 math reasoning (integer answers in \\boxed{})",
            "task_type": "math_reasoning",
            "supports_few_shot": False,
            "typical_input_length": "50-300 tokens",
            "typical_output_length": "1-20 tokens"
        },
        DatasetFormat.GOVREPORT: {
            "description": "GovReport long-document summarization (local JSONL)",
            "task_type": "summarization",
            "supports_few_shot": False,
            "typical_input_length": "long (up to ~16K context)",
            "typical_output_length": "variable"
        },
        DatasetFormat.PG19: {
            "description": "PG19 long-context dataset (local JSONL)",
            "task_type": "long_context",
            "supports_few_shot": False,
            "typical_input_length": "long (up to ~16K context)",
            "typical_output_length": "variable"
        },
        DatasetFormat.BOOKSUM: {
            "description": "BookSum long-context summarization (local JSONL)",
            "task_type": "summarization",
            "supports_few_shot": False,
            "typical_input_length": "long (up to ~16K context)",
            "typical_output_length": "variable"
        },
        DatasetFormat.GPQA: {
            "description": "GPQA difficult multiple-choice QA",
            "task_type": "multiple_choice",
            "supports_few_shot": False,
            "typical_input_length": "100-300 tokens",
            "typical_output_length": "multiple choice answer"
        },
        DatasetFormat.MMLU_PRO: {
            "description": "MMLU-Pro advanced reasoning/knowledge dataset",
            "task_type": "multiple_choice",
            "supports_few_shot": False,
            "typical_input_length": "variable",
            "typical_output_length": "multiple choice answer"
        }
    }

    return info.get(dataset_format, {"description": "Unknown dataset format"})


# Backward compatibility functions
def build_prompt(example: dict, dataset_name: str) -> str:
    """Legacy function for backward compatibility."""
    if dataset_name == "cnn_dm":
        article = example.get("article") or example.get("text") or ""
        return f"Summarize the following article.\n\nArticle:\n{article}\n\nSummary:"
    if dataset_name == "gsm8k":
        q = example.get("question") or example.get("prompt") or ""
        return f"Q: {q}\nA:"
    return example.get("text") or example.get("content") or str(example)


def load_dataset_convenience(dataset: str, num_samples: int):
    """Legacy function for backward compatibility."""
    if dataset == "cnn_dm":
        return get_dataset(DatasetFormat.CNN_DM_SUMMARIZATION, num_samples=num_samples)
    elif dataset == "gsm8k":
        return get_dataset(DatasetFormat.GSM8K, num_samples=num_samples)
    else:
        raise ValueError("Use get_dataset() function with DatasetFormat for new code")