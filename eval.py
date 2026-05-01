import argparse
import json
import os
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Union

import openai
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeRemainingColumn)
from rich.table import Table

from paths import ensure_dir, resolve_in_output_root, resolve_output_path

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
                                
from dotenv import load_dotenv
load_dotenv()

ExtractionConfigType = Union[ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig]

CORRECTNESS_PROMPT_TEMPLATE = """
Human:
You are a teacher grading a quiz.
You are given a question, the student's answer (this may be truncated to the last 1000 tokens), and the true answer, and are asked to score the student answer as either Correct or Incorrect.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: Correct or Incorrect here

Grade the student answer's based ONLY on choice accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer. If the student fails to answer, mark it as incorrect. Begin!
If the extracted answer is the same as the true answer mark it as correct, even if some intermediate steps are wrong. Only the final answer matters.

QUESTION: \n<question>{query}</question>\n
STUDENT ANSWER: \n<student_answer>{answer}</student_answer>\n
TRUE ANSWER: \n<expected_answer>{expected_answer}</expected_answer>\n
GRADE:

Your response should be in JSON format as follows:
{{
    "extracted_answer": "(The final answer extracted from the student's response, e.g., 'A', '42', 'photosynthesis', etc.)"
    "grade": "(correct or incorrect. Mark as correct if the final answer is the same as the true answer, even if some intermediate steps are wrong. Only the final answer matters.)",
    "justification": "(Without mentioning the student/teacher framing of this prompt, explain why the STUDENT ANSWER is Correct or Incorrect. Use one or two sentences maximum. Keep the answer as concise as possible.)",
}}
Assistant:
"""

DATASET_CONFIGS = {
    "gsm8k": {
        "eval_type": "Math",
        "gold_config": [ExprExtractionConfig(try_extract_without_anchor=True)],
        "pred_config": [LatexExtractionConfig(), ExprExtractionConfig()],
        "response_field": "response",
    },
    "aime": {
        "eval_type": "Math",
        "gold_config": [ExprExtractionConfig(try_extract_without_anchor=True)],
        "pred_config": [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        "response_field": "response",
    },
    "cfa-l1": {
        "eval_type": "CFA",
        "gold_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True, try_extract_without_anchor=True)],
        "pred_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True)],
        "gold_preprocessor": lambda ans: ans.split('_')[-1].upper() if '_' in ans else ans.upper(),
        "response_field": "response",
    },
    "ablation": {
        "eval_type": "Mixed",
        "response_field": "response",
        "sub_configs": {
            "l1": {
                "eval_type": "CFA",
                "gold_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True, try_extract_without_anchor=True)],
                "pred_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True)],
                "gold_preprocessor": lambda ans: ans.split('_')[-1].upper() if '_' in ans else ans.upper(),
            },
            "l2": {
                "eval_type": "CFA",
                "gold_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True, try_extract_without_anchor=True)],
                "pred_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True)],
                "gold_preprocessor": lambda ans: ans.split('_')[-1].upper() if '_' in ans else ans.upper(),
            },
            "gsm": {
                "eval_type": "Math",
                "gold_config": [ExprExtractionConfig(try_extract_without_anchor=True)],
                "pred_config": [LatexExtractionConfig(), ExprExtractionConfig()],
            },
            "math": {
                "eval_type": "Math",
                "gold_config": [ExprExtractionConfig(try_extract_without_anchor=True)],
                "pred_config": [LatexExtractionConfig(), ExprExtractionConfig()],
            },
            "aime": {
                "eval_type": "Math",
                "gold_config": [ExprExtractionConfig(try_extract_without_anchor=True)],
                "pred_config": [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
            },
        }
    },
}
DATASET_CONFIGS["math500"] = DATASET_CONFIGS["gsm8k"]
DATASET_CONFIGS["cfa-l2"] = DATASET_CONFIGS["cfa-l1"]

def validate_correctness_response(response: str) -> Dict[str, str]:
    """
    Validate and parse the correctness response from the LLM.
    Handles responses wrapped in markdown code blocks and falls back to regex.
    """
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in the response.")
        
        json_str = json_match.group(0)
        result = json.loads(json_str)
        
        if not isinstance(result, dict):
            raise ValueError("Parsed JSON is not a dictionary.")

        return {
            'grade': str(result.get('grade', '')).lower().strip(),
            'justification': str(result.get('justification', '')),
            'extracted_answer': str(result.get('extracted_answer', ''))
        }
    except (json.JSONDecodeError, ValueError) as e:
        # use regex as a fallback to extract the fields from the original response.
        grade_pattern = r'"grade":\s*"([^"]+)"'
        justification_pattern = r'"justification":\s*"([^"]*(?:[^"\\]|\\.)*)"'
        extracted_answer_pattern = r'"extracted_answer":\s*"([^"]*(?:[^"\\]|\\.)*)"'
        
        grade_match = re.search(grade_pattern, response, re.IGNORECASE | re.DOTALL)
        justification_match = re.search(justification_pattern, response, re.IGNORECASE | re.DOTALL)
        extracted_answer_match = re.search(extracted_answer_pattern, response, re.IGNORECASE | re.DOTALL)
        
        return {
            'grade': grade_match.group(1).lower().strip() if grade_match else 'incorrect',
            'justification': justification_match.group(1).strip() if justification_match else f'Parsing error: {str(e)}',
            'extracted_answer': extracted_answer_match.group(1).strip() if extracted_answer_match else ''
        }

def extract_student_answer(response: str, dataset_type: str = "") -> str:
    """Extract the final answer from a student's response using heuristics."""
    if not response:
        return ""
    
    response = response.strip()
    
    # For CFA/multiple choice questions, look for A, B, C patterns
    if dataset_type.lower() == "cfa":
        # Look for patterns like "Answer: A", "The answer is B", "(C)", "A)", etc.
        match = re.search(r'(?i)(?:the\s+answer\s+is|final\s+answer\s*:|choice\s+is|is:)\s*\b([A-C])\b|(?:\(|\[)\s*\b([A-C])\b(?:\)|\])|\b([A-C])\b(?=\s*is\s+correct)', response)
        if match:
            return next(g for g in match.groups() if g is not None).upper()
        # Fallback to simpler patterns if the above fails
        final_char_match = re.search(r'\b([A-C])\s*$', response)
        if final_char_match:
            return final_char_match.group(1).upper()
    
    # For math questions, look for boxed answers or "final answer" markers.
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    final_answer_match = re.search(r'(?i)(?:the\s+final\s+answer\s+is|my\s+answer\s+is|the\s+answer\s+is|result\s+is)\s*:?\s*(.*?)(?:\.|\n|Therefore|So|$)', response)
    if final_answer_match:
        answer = final_answer_match.group(1).strip()
        # Remove trailing punctuation
        return re.sub(r'[\s\.,;!]+$', '', answer)

    return ""

def setup_openai() -> openai.AsyncOpenAI:
    """Configures and returns an async OpenAI client for OpenRouter."""
    return openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

async def grade_with_llm_async(client: openai.AsyncOpenAI, model_name: str, question: str, student_response: str, correct_answer: str, dataset_type: str = "") -> Dict[str, Any]:
    """Asynchronously grades a response using an LLM."""
    try:
        prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
            query=question,
            answer=student_response,
            expected_answer=correct_answer
        )
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0
        )
        
        content = response.choices[0].message.content or ""
        parsed_result = validate_correctness_response(content)
        
        is_correct = parsed_result.get("grade") == "correct"
        
        # If LLM didn't extract an answer or extracted a long, non-answer string,
        # use our more reliable regex-based extractor as a fallback.
        final_extracted_answer = parsed_result.get("extracted_answer", "")
        if not final_extracted_answer or len(final_extracted_answer) > 50:
            final_extracted_answer = extract_student_answer(student_response, dataset_type)
        
        return {
            "is_correct": is_correct,
            "justification": parsed_result.get("justification", ""),
            "extracted_answer": final_extracted_answer,
            "evaluation_prompt": prompt,
            "evaluation_response": content
        }
            
    except Exception as e:
        # If the API call fails, return error details
        fallback_answer = extract_student_answer(student_response, dataset_type)
        return {
            "is_correct": False,
            "justification": f"LLM_GRADING_ERROR: {str(e)}",
            "extracted_answer": fallback_answer,
            "evaluation_prompt": prompt if 'prompt' in locals() else "",
            "evaluation_response": f"ERROR: {str(e)}"
        }

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model outputs using either math-verify parser or LLM grading.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to the directory with result files (*.json).")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CONFIGS.keys(), help="Name of the dataset being evaluated.")
    parser.add_argument("--output-file", type=Path, default=None, help="Path to save the detailed evaluation JSON file. Defaults to <input-dir>/evaluation/evaluation_summary.json.")
    parser.add_argument("--mode", type=str, required=True, choices=["llm", "parser"], help="Evaluation mode: 'llm' for LLM-based grading, 'parser' for math-verify/string matching.")
    parser.add_argument("--llm-model", type=str, default="google/gemini-2.5-flash", help="Model for LLM grading (only used in 'llm' mode).")
    parser.add_argument("--num-workers", type=int, default=100, help="Number of concurrent LLM calls (only used in 'llm' mode)")
    return parser.parse_args()

def serialize_object(obj: Any) -> str:
    """Convert a parsed object (e.g., from sympy) to a string for JSON."""
    if obj is None: return ""
    if isinstance(obj, list) and not obj: 
        return "[]"
    if isinstance(obj, list): 
        return ", ".join(map(str, obj))
    return str(obj)

def get_question_from_prompt(item_data: Dict) -> str:
    """Extract the user question from various possible prompt structures."""
    # Handle the new structure with direct "question" field
    if "question" in item_data and isinstance(item_data["question"], str):
        return item_data["question"]
    
    # Handle existing prompt-based structures
    full_prompt = item_data.get("full_prompt")
    if isinstance(full_prompt, list):
        for message in reversed(full_prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if content and isinstance(content, str):
                    return content
    return str(item_data.get("prompt_question", ""))

def get_ground_truth_answer(item_data: Dict) -> str:
    """Extract the ground truth answer from various possible structures."""
    # Handle the new structure with "answer_ground_truth" field
    if "answer_ground_truth" in item_data:
        return str(item_data["answer_ground_truth"])
    
    # Handle existing structures
    return str(item_data.get("expected_answer", item_data.get("gold_answer", item_data.get("answer", ""))))

def get_model_responses(item_data: Dict) -> Dict[str, str]:
    """Extract model responses from various possible structures."""
    responses = {"standard": "", "entropix": ""}
    
    # Handle the new structure with "response_model" field
    if "response_model" in item_data:
        response_content = item_data["response_model"]
        if isinstance(response_content, str):
            # If it's a single response, treat it as standard
            responses["standard"] = response_content
        elif isinstance(response_content, dict):
            # If it's a dict, look for specific keys
            responses["standard"] = str(response_content.get("response", response_content.get("content", "")))
            # Check if there are multiple responses or entropix-specific response
            if "entropix" in response_content:
                responses["entropix"] = str(response_content["entropix"])
    
    # Handle existing structures (fallback)
    if not responses["standard"]:
        std_data = item_data.get("gen_data", item_data.get("standard", {}))
        responses["standard"] = str(std_data.get("response", "")) if isinstance(std_data, dict) else ""
    
    if not responses["entropix"]:
        ent_data = item_data.get("entropix", {})
        responses["entropix"] = str(ent_data.get("response", "")) if isinstance(ent_data, dict) else ""
    
    return responses

def get_question_id(item_data: Dict, index: int) -> str:
    """Extract question ID from various possible structures."""
    # Handle the new structure
    if "question_id" in item_data:
        return str(item_data["question_id"]).replace("/", "_")
    
    # Handle exam-based ID structure
    if "exam" in item_data and "question_number" in item_data:
        exam = item_data["exam"]
        q_num = item_data["question_number"]
        return f"{exam}_q{q_num}".replace("/", "_")
    
    # Handle existing structures
    if "id" in item_data:
        return str(item_data["id"]).replace("/", "_")
    
    # Fallback to index-based ID
    return f"item_{index}"

def get_ablation_config(item_data: Dict, base_config: Dict) -> Dict:
    """Get the appropriate sub-config for an ablation question based on original exam type."""
    # Check both 'original_exam_type' (from run.py output) and 'exam_type' (from raw ablation data)
    original_exam_type = item_data.get("original_exam_type") or item_data.get("exam_type")
    if not original_exam_type:
        raise ValueError(f"Missing original_exam_type/exam_type for ablation question {item_data.get('question_id', 'unknown')}")
    
    sub_config = base_config["sub_configs"].get(original_exam_type)
    if not sub_config:
        raise ValueError(f"Unknown original exam type '{original_exam_type}' for ablation question {item_data.get('question_id', 'unknown')}")
    
    return sub_config

def evaluate_with_parser(response: str, gold_answer: str, config: Dict, item_data: Dict = None) -> Dict[str, Any]:
    """Evaluate a response using the math-verify parser."""
    # Handle ablation dataset by getting the appropriate sub-config
    if config.get("eval_type") == "Mixed" and item_data:
        actual_config = get_ablation_config(item_data, config)
    else:
        actual_config = config
    
    # Parse gold answer
    parsed_gold = parse(gold_answer, extraction_config=actual_config["gold_config"], extraction_mode="any_match", parsing_timeout=5)
    
    # Parse prediction
    parsed_pred = parse(response, extraction_config=actual_config["pred_config"], parsing_timeout=5)
    
    # Verify correctness
    is_correct = verify(parsed_gold, parsed_pred)
    
    return {
        "correct": is_correct,
        "parsed_gold": serialize_object(parsed_gold),
        "parsed_prediction": serialize_object(parsed_pred),
        "extracted_answer": serialize_object(parsed_pred),
        "evaluation_details": {
            "mode": "parser",
            "parser_type": actual_config["eval_type"]
        }
    }

async def evaluate_with_llm(client: openai.AsyncOpenAI, model_name: str, question: str, response: str, gold_answer: str, dataset_type: str, item_data: Dict = None) -> Dict[str, Any]:
    """Evaluate a response using LLM grading."""
    # For ablation dataset, determine the actual dataset type from the original exam type
    if dataset_type == "Mixed" and item_data:
        original_exam_type = item_data.get("original_exam_type") or item_data.get("exam_type")
        if original_exam_type in ("l1", "l2"):
            actual_dataset_type = "CFA"
        elif original_exam_type in ("gsm", "math", "aime"):
            actual_dataset_type = "Math"
        else:
            raise ValueError(f"Unknown original exam type '{original_exam_type}' for ablation question")
    else:
        actual_dataset_type = dataset_type
    
    result = await grade_with_llm_async(client, model_name, question, response, gold_answer, actual_dataset_type)
    
    return {
        "correct": result["is_correct"],
        "extracted_answer": result["extracted_answer"],
        "justification": result["justification"],
        "evaluation_details": {
            "mode": "llm",
            "model": model_name,
            "evaluation_prompt": result["evaluation_prompt"],
            "evaluation_response": result["evaluation_response"]
        }
    }

async def run_evaluation(args: argparse.Namespace, console: Console, all_results: List[Dict], config: Dict, output_dir: Path):
    error_dir = output_dir / "error_analysis"
    error_dir.mkdir(exist_ok=True, parents=True)
    
    evaluation_data = []
    
    if args.mode == "parser":
        console.print(f"[cyan]Running parser-based evaluation ({config['eval_type']})...[/cyan]")
        
        for i, item in enumerate(all_results):
            try:
                # Preprocess gold answer
                gold_str_raw = get_ground_truth_answer(item)
                gold_preprocessor = config.get("gold_preprocessor", lambda x: x)
                # # Preprocess gold answer based on dataset type
                # gold_str_raw = item.get("expected_answer", item.get("gold_answer", item.get("answer", "")))
                # 
                # # For ablation datasets, get the appropriate preprocessor
                if config.get("eval_type") == "Mixed":
                    actual_config = get_ablation_config(item, config)
                    gold_preprocessor = actual_config.get("gold_preprocessor", lambda x: x)
                else:
                    gold_preprocessor = config.get("gold_preprocessor", lambda x: x)
                
                gold_str = gold_preprocessor(str(gold_str_raw))
                
                # Get responses
                std_response = get_model_responses(item)["standard"]
                ent_response = get_model_responses(item)["entropix"]
                
                # Evaluate both methods
                std_result = evaluate_with_parser(std_response, gold_str, config, item) if std_response else {"correct": False, "evaluation_details": {"mode": "parser", "error": "No response"}}
                ent_result = evaluate_with_parser(ent_response, gold_str, config, item) if ent_response else {"correct": False, "evaluation_details": {"mode": "parser", "error": "No response"}}
                
                result_details = {
                    "question_id": get_question_id(item, i),
                    "question": get_question_from_prompt(item),
                    "gold_answer": gold_str,
                    "original_exam_type": item.get("original_exam_type") or item.get("exam_type"),
                    "standard": {
                        "response": std_response,
                        "correct": std_result["correct"],
                        "extracted_answer": std_result.get("extracted_answer", ""),
                        "evaluation_details": std_result["evaluation_details"]
                    },
                    "entropix": {
                        "response": ent_response,
                        "correct": ent_result["correct"],
                        "extracted_answer": ent_result.get("extracted_answer", ""),
                        "evaluation_details": ent_result["evaluation_details"]
                    }
                }
                evaluation_data.append(result_details)
                
            except Exception as e:
                console.print(f"[red]Error processing item {item.get('id', i)}: {e}[/red]")
    
    else:  # args.mode == "llm"
        console.print("[cyan]Running LLM-based evaluation...[/cyan]")
        
        if "OPENROUTER_API_KEY" not in os.environ:
            console.print("[bold red]Error: LLM mode requires the OPENROUTER_API_KEY environment variable.[/bold red]")
            return []
        
        client = setup_openai()
        llm_tasks = []
        
        # Prepare all tasks
        for i, item in enumerate(all_results):
            try:
                # Preprocess gold answer
                gold_str_raw = get_ground_truth_answer(item)
                gold_preprocessor = config.get("gold_preprocessor", lambda x: x)
                
                # For ablation datasets, get the appropriate preprocessor
                if config.get("eval_type") == "Mixed":
                    actual_config = get_ablation_config(item, config)
                    gold_preprocessor = actual_config.get("gold_preprocessor", lambda x: x)
                else:
                    gold_preprocessor = config.get("gold_preprocessor", lambda x: x)
                
                gold_str = gold_preprocessor(str(gold_str_raw))
                
                question = get_question_from_prompt(item)
                std_response = get_model_responses(item)["standard"]
                ent_response = get_model_responses(item)["entropix"]
                
                item_info = {
                    "index": i,
                    "question_id": get_question_id(item, i),
                    "question": question,
                    "gold_answer": gold_str,
                    "std_response": std_response,
                    "ent_response": ent_response,
                    "original_exam_type": item.get("original_exam_type") or item.get("exam_type"),
                    "item_data": item
                }
                
                # Create tasks for both methods if responses exist
                if std_response:
                    task = evaluate_with_llm(client, args.llm_model, question, std_response[-4000:], gold_str, config.get("eval_type", ""), item)
                    llm_tasks.append({"item_info": item_info, "method": "standard", "task": task})
                
                if ent_response:
                    task = evaluate_with_llm(client, args.llm_model, question, ent_response[-4000:], gold_str, config.get("eval_type", ""), item)
                    llm_tasks.append({"item_info": item_info, "method": "entropix", "task": task})
                    
            except Exception as e:
                console.print(f"[red]Error preparing item {item.get('id', i)}: {e}[/red]")
        
        if llm_tasks:
            console.print(f"[cyan]Grading {len(llm_tasks)} responses with LLM (workers={args.num_workers})...[/cyan]")
            
            # Process tasks concurrently
            sem = asyncio.Semaphore(args.num_workers)
            
            async def run_with_semaphore(task):
                async with sem:
                    return await task
            
            all_tasks = [entry['task'] for entry in llm_tasks]
            
            async def run_and_update_progress(task):
                result = await run_with_semaphore(task)
                progress.update(progress_task, advance=1)
                return result
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console) as progress:
                progress_task = progress.add_task("Grading with LLM...", total=len(all_tasks))
                results = await asyncio.gather(*[run_and_update_progress(task) for task in all_tasks])
            
            # Group results by item
            items_map = {}
            for entry, result in zip(llm_tasks, results):
                item_info = entry['item_info']
                method = entry['method']
                idx = item_info['index']
                
                if idx not in items_map:
                    items_map[idx] = {
                        "question_id": item_info['question_id'],
                        "question": item_info['question'],
                        "gold_answer": item_info['gold_answer'],
                        "original_exam_type": item_info['original_exam_type'],
                        "standard": {"response": item_info['std_response'], "correct": False, "evaluation_details": {"mode": "llm", "error": "default: Missing response"}},
                        "entropix": {"response": item_info['ent_response'], "correct": False, "evaluation_details": {"mode": "llm", "error": "default: Missing response"}}
                    }
                
                items_map[idx][method] = {
                    "response": item_info['std_response'] if method == 'standard' else item_info['ent_response'],
                    "correct": result["correct"] or (item_info["gold_answer"].lower() == result["extracted_answer"].lower()),
                    "extracted_answer": result["extracted_answer"],
                    "justification": result.get("justification", ""),
                    "evaluation_details": result["evaluation_details"]
                }
            
            # Convert to list
            evaluation_data = [items_map[i] for i in sorted(items_map.keys())]
    
    # Count correct answers
    correct_counts = {"standard": 0, "entropix": 0}
    total_counts = {"standard": 0, "entropix": 0}
    
    for result in evaluation_data:
        for method in ["standard", "entropix"]:
            if result[method]["response"]:
                total_counts[method] += 1
                if result[method]["correct"]:
                    correct_counts[method] += 1
        
        # Save error analysis for incorrect responses
        std_incorrect = result["standard"]["response"] and not result["standard"]["correct"]
        ent_incorrect = result["entropix"]["response"] and not result["entropix"]["correct"]
        if std_incorrect or ent_incorrect:
            with open(error_dir / f"error_q{result['question_id']}.json", "w", encoding='utf-8') as f_err:
                json.dump(result, f_err, indent=2, ensure_ascii=False)
    
    # Sort by question ID
    evaluation_data.sort(key=lambda x: x["question_id"])
    
    # Display summary
    display_summary_table(console, args.mode, config['eval_type'], args.dataset, correct_counts, total_counts, evaluation_data)
    
    return evaluation_data

def display_summary_table(console: Console, mode: str, eval_type: str, dataset_name: str, correct_counts: dict, total_counts: dict, evaluation_data: List[Dict] = None):
    """Creates and prints a summary table for the evaluation results."""
    mode_display = "LLM" if mode == "llm" else f"{eval_type} Parser"
    title = f"Evaluation Summary for {dataset_name.upper()} (Mode: {mode_display})"

    summary = Table(title=title, header_style="bold magenta", show_header=True)
    summary.add_column("Method", style="cyan", no_wrap=True)
    summary.add_column("Correct", style="green")
    summary.add_column("Total", style="blue")
    summary.add_column("Accuracy", style="yellow")

    for method in ["standard", "entropix"]:
        if method == "entropix": 
            summary.add_section()
        
        method_name_display = method.capitalize()
        total = total_counts[method]
        correct = correct_counts[method]
        
        if total == 0:
            summary.add_row(f"{method_name_display}", "0", "0", "N/A")
        else:
            accuracy = (correct / total) * 100
            summary.add_row(f"{method_name_display}", str(correct), str(total), f"{accuracy:.2f}%")

    console.print(summary)
    
    # For ablation dataset, show breakdown by exam type
    if dataset_name == "ablation" and evaluation_data:
        console.print("\n")
        
        # Collect stats by exam type
        exam_type_stats = {}
        for result in evaluation_data:
            exam_type = None
            
            # First try to get exam type from the original data structure 
            exam_type = result.get("original_exam_type")
            
            # Fall back to extracting from question_id patterns like "l1_q1", "gsm_q5", etc.
            if not exam_type:
                question_id = result.get("question_id", "")
                if "_" in question_id:
                    exam_type = question_id.split("_")[0]
            
            if exam_type and exam_type in ["l1", "l2", "gsm", "math", "aime"]:
                if exam_type not in exam_type_stats:
                    exam_type_stats[exam_type] = {
                        "standard": {"correct": 0, "total": 0},
                        "entropix": {"correct": 0, "total": 0}
                    }
                
                for method in ["standard", "entropix"]:
                    if result[method]["response"]:
                        exam_type_stats[exam_type][method]["total"] += 1
                        if result[method]["correct"]:
                            exam_type_stats[exam_type][method]["correct"] += 1
        
        if exam_type_stats:
            breakdown_table = Table(title="Ablation Dataset Breakdown by Exam Type", header_style="bold magenta", show_header=True)
            breakdown_table.add_column("Exam Type", style="cyan", no_wrap=True)
            breakdown_table.add_column("Method", style="white", no_wrap=True)
            breakdown_table.add_column("Correct", style="green")
            breakdown_table.add_column("Total", style="blue")
            breakdown_table.add_column("Accuracy", style="yellow")
            
            for exam_type in sorted(exam_type_stats.keys()):
                for i, method in enumerate(["standard", "entropix"]):
                    stats = exam_type_stats[exam_type][method]
                    total = stats["total"]
                    correct = stats["correct"]
                    
                    exam_display = exam_type.upper() if i == 0 else ""
                    method_display = method.capitalize()
                    
                    if total == 0:
                        breakdown_table.add_row(exam_display, method_display, "0", "0", "N/A")
                    else:
                        accuracy = (correct / total) * 100
                        breakdown_table.add_row(exam_display, method_display, str(correct), str(total), f"{accuracy:.2f}%")
                
                if exam_type != sorted(exam_type_stats.keys())[-1]:
                    breakdown_table.add_section()
            
            console.print(breakdown_table)

async def main():
    args = _parse_args()
    console = Console()

    evaluated_input_dir = resolve_in_output_root(args.input_dir)
    args.input_dir = evaluated_input_dir

    console.print(f"[bold blue]Evaluation mode: {args.mode.upper()}[/bold blue]")
    
    console.print(f"[cyan]Scanning for result files (*.json) in '{evaluated_input_dir}'...[/cyan]")
    json_files = sorted(list(evaluated_input_dir.glob("*.json")), key=lambda p: p.stem)
    if not json_files:
        console.print(f"[red]Error: No result files found in {args.input_dir}. Exiting.[/red]")
        return
    console.print(f"[green]Found {len(json_files)} result files. Loading...[/green]")

    all_results = []
    for p in json_files:
        try:
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
                data['id'] = p.stem
                all_results.append(data)
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not decode JSON from {p.name}. Skipping.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read file {p.name} due to {e}. Skipping.[/yellow]")

    if not all_results:
        console.print("[red]Error: No valid result files could be loaded. Exiting.[/red]")
        return

    dataset_config = DATASET_CONFIGS[args.dataset]
    console.print(f"[cyan]Using configuration for dataset: '{args.dataset}' (Type: {dataset_config['eval_type']})[/cyan]")

    if args.output_file:
        output_path = resolve_output_path(args.output_file, f"evaluation_{args.mode}/evaluation_summary.json")
    else:
        output_path = evaluated_input_dir / f"evaluation_{args.mode}" / "evaluation_summary.json"
        ensure_dir(output_path.parent)
    
    evaluation_data = await run_evaluation(args, console, all_results, dataset_config, output_path.parent)

    if evaluation_data:
        with open(output_path, "w", encoding='utf-8') as f_out:
            json.dump(evaluation_data, f_out, indent=2, ensure_ascii=False)
        console.print(f"\n[bold green]Detailed evaluation report saved to:[/bold green] {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
