import argparse
import json
import os
from datetime import datetime
import sys
import pandas as pd
from rich import print as rprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
from confidence_beam_search_vers import reasoning_confidence_beam_search
from beam_search_vers import reasoning_beam_search, setup_models

from paths import resolve_output_path

os.environ["HF_HOME"] = "/home/jasonxli/links/scratch/huggingface"  # Change this to your desired cache path
os.environ["HF_MODULES_CACHE"] = "/home/jasonxli/links/scratch/huggingface/modules"  # Change this to your desired cache path

p = os.environ["HF_MODULES_CACHE"]
sys.path.insert(0, p) if p not in sys.path else None

def parse_args():
    """
    Parse command-line arguments.
    Arguments from the original `run.py` are kept for data handling.
    Arguments for `entropix` are replaced with ones for `reasoning_beam_search`.
    """
    parser = argparse.ArgumentParser(description="Run Reasoning Beam Search with a PRM Scorer")

    # --- Model Selection ---
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Generator model name (e.g., Qwen/Qwen3-1.7B, meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--scorer-model", type=str, default=None, help="PRM Scorer model name or local path (default: cached Qwen2.5-Math-PRM-7B)")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization to save memory.")
    # --- Reasoning Beam Search Parameters ---
    parser.add_argument("--num-beams", type=int, default=4, help="Number of main beams to maintain in the search.")
    parser.add_argument("--num-expansions", type=int, default=4, help="Number of candidate branches to generate for each active beam.")
    parser.add_argument("--num-steps", type=int, default=100, help="Maximum number of reasoning steps (expansions) to perform.")
    parser.add_argument("--max-step-tokens", type=int, default=1024, help="Maximum number of new tokens to generate in a single step.")
    parser.add_argument("--entropy-threshold", type=float, default=1.5, help="Entropy threshold for confidence-based pruning (0.0 to disable).")
    parser.add_argument("--confidence-beam-search", action="store_true", help="Use confidence-based beam search instead of standard beam search.")
    
    # --- Data and Execution Parameters ---
    parser.add_argument("--exam", type=str, default="gsm", choices=["l1", "l2", "l3", "gsm", "math", "aime"], help="Exam dataset to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process (default: 5).")
    parser.add_argument("--start", type=int, default=1, help="Start from question number (1-indexed).")

    # --- Output Parameters ---
    parser.add_argument("-o", "--out", type=str, default=None, help="Output directory for all results (default: {model}_{exam}_{timestamp}).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Print less output during generation.")

    return parser.parse_args()

def _dataset_to_dataframe(dataset, preferred_split=None):
    """Normalize a Hugging Face dataset or dataset dict into a DataFrame."""
    if isinstance(dataset, DatasetDict):
        if preferred_split and preferred_split in dataset:
            dataset = dataset[preferred_split]
        else:
            for split_name in ("test", "validation", "train"):
                if split_name in dataset:
                    dataset = dataset[split_name]
                    break
            else:
                dataset = next(iter(dataset.values()))
    if hasattr(dataset, "to_pandas"):
        try:
            return dataset.to_pandas()
        except Exception:
            return pd.DataFrame(dataset)
    return pd.DataFrame(dataset)

def load_exam_data(exam_type, limit=None, start=1):
    
    """Load exam data based on the specified type."""
    if exam_type == "l1":
        df = pd.read_json("data/l1_exams.json")
    elif exam_type == "l2":
        df = pd.read_json("data/l2_exams.json")
    elif exam_type == "l3":
        df = pd.read_json("data/l3_exams.json")
    elif exam_type == "gsm":
        dataset = load_from_disk("./dataset_cache/gsm8k")
        df = _dataset_to_dataframe(dataset)
    elif exam_type == "math":
        dataset = load_from_disk("./dataset_cache/math")
        df = _dataset_to_dataframe(dataset)
        df = df.rename(columns={"problem": "question"})
    elif exam_type == "aime":
        dataset = load_from_disk("./dataset_cache/aime")
        df = _dataset_to_dataframe(dataset)
        df = df.rename(columns={"problem": "question"})
    else:
        raise ValueError(f"Unknown exam type: {exam_type}")

    # Apply start and limit
    start_idx = start - 1  # Convert to 0-indexed
    if limit:
        end_idx = start_idx + limit
        return df.iloc[start_idx:end_idx]
    else:
        return df.iloc[start_idx:]

def format_prompt(exam_type, question_data):
    """Format the prompt based on the exam type."""
    if exam_type in ("l1", "l2", "l3"):
        messages = [
            {
                "role": "system", "content": """You are an expert financial analyst.
You are given questions about various financial topics, from quantitative analysis to portfolio management to ethics of being a chartered financial analyst (CFA).
Each question includes 3 potential answers, A B and C, one of which is correct (or in some cases, more correct than the others).
Indicate the correct answer: A, B, or C."""
            }
        ]
        if exam_type == "l1":
            messages.append(
                {
                    "role": "user",
                    "content": f"""{question_data.question}
A. {question_data.choice_a}
B. {question_data.choice_b}
C. {question_data.choice_c}""",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"""{question_data.case}

{question_data.question}
A. {question_data.choice_a}
B. {question_data.choice_b}
C. {question_data.choice_c}""",
                }
            )

    elif exam_type in ("gsm", "math", "aime"):
        messages = [
            # {"role": "system", "content": "You are a mathematical expert."},
            {"role": "user", "content": question_data.question},
        ]
    else:
        raise ValueError(f"Unknown exam type: {exam_type}")

    return messages


def main():
    args = parse_args()
    if args.confidence_beam_search:
        rprint(f"[bold magenta]Using Confidence-Based Reasoning Beam Search with entropy threshold {args.entropy_threshold} with model {args.model} on exam {args.exam}[/bold magenta]")
    else:
        rprint(f"[bold green]Starting Reasoning Beam Search with model {args.model} on exam {args.exam}[/bold green]")

    # Load exam data based on the selected type
    df = load_exam_data(args.exam, args.limit, args.start)

    print(df)

    # Load models
    generator_model, generator_tokenizer, score_model, score_tokenizer, device = setup_models(args.model, args.scorer_model)

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "_")
    default_leaf = f"results_{model_safe}_{args.exam}_{timestamp}"
    output_dir = resolve_output_path(args.out, default_leaf)

    rprint(f"Saving all results to [bold yellow]{str(output_dir)}/[/bold yellow]")

 # --- 2. Main Loop ---
    for idx, q in df.iterrows():
        # The original `df.iterrows()` gives the DataFrame index, not the 1-based question number
        # Let's use `start` and a counter for a more intuitive question number
        question_num = args.start + (df.index.get_loc(idx))
        
        rprint(f"\n[bold]Processing Question {question_num}/{args.start + len(df) - 1}[/bold]")
        
        messages = format_prompt(args.exam, q)
        if not args.quiet:
            rprint(f"[bold cyan]Prompt:[/bold cyan]\n{messages[-1]['content']}")

        try:
            # --- 3. Run Generation ---
            # This is the core change: calling the new function
            if args.confidence_beam_search:
                best_beam = reasoning_confidence_beam_search(
                    messages=messages,
                    generator_model=generator_model,
                    generator_tokenizer=generator_tokenizer,
                    score_model=score_model,
                    score_tokenizer=score_tokenizer,
                    device=device,
                    num_main_beams=args.num_beams,
                    num_expansions_per_beam=args.num_expansions,
                    num_reasoning_steps=args.num_steps,
                    max_step_tokens=args.max_step_tokens,
                    entropy_threshold=args.entropy_threshold,
                )
            else:
                best_beam = reasoning_beam_search(
                    messages=messages,
                    generator_model=generator_model,
                    generator_tokenizer=generator_tokenizer,
                    score_model=score_model,
                    score_tokenizer=score_tokenizer,
                    device=device,
                    num_main_beams=args.num_beams,
                    num_expansions_per_beam=args.num_expansions,
                    num_reasoning_steps=args.num_steps,
                    max_step_tokens=args.max_step_tokens,
                )

            response_text = best_beam['text'].replace(messages[0]['content'], '').strip() # Clean up prompt from response
            response_text = response_text.replace(messages[-1]['content'], '').strip()

            trace = best_beam.get('trace', {})


            # --- 4. Save Results ---
            result = {
                "exam": args.exam,
                "question_number": question_num,
                "question": messages[-1]['content'],
                "answer_ground_truth": q.answer,
                "response_model": response_text,
                "generation_details": { # Storing the full details from the beam search
                    "final_score": best_beam["step_score"],
                    "trace": trace,
                }
            }

            # Add dataset-specific metadata
            if args.exam in ("l1", "l2", "l3"):
                result.update({"question_id": q.id, "explanation": q.explanation})
            elif args.exam == "math":
                 result.update({"question_id": q.unique_id, "solution": q.solution})
            elif args.exam == "aime":
                 result.update({"question_id": q.id, "solution": q.solution})
            elif args.exam == "gsm":
                result.update({"question_id": idx, "solution": q.answer}) # GSM answer is the solution

            # Save individual question result
            question_file = output_dir / f"q{question_num}.json"
            with question_file.open('w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Print a summary
            q_text = result["question"].strip().replace("\n", " ")
            truncated_q = (q_text[:150] + "...") if len(q_text) > 150 else q_text
            resp_text = result["response_model"].strip().replace("\n", " ")
            truncated_resp = (resp_text[:150] + "...") if len(resp_text) > 150 else resp_text

            rprint(f"[bold cyan]Q:[/bold cyan] {truncated_q}")
            rprint(f"[bold green]Model:[/bold green] {truncated_resp}")
            rprint(f"[bold yellow]Answer:[/bold yellow] {result['answer_ground_truth']}\n")
            rprint(f"[grey50]Saved to {question_file}[/grey50]")

        except Exception as e:
            rprint(f"[bold red]Error processing question {question_num}: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            error_result = {
                "exam": args.exam,
                "question_number": question_num,
                "question": messages[-1]['content'],
                "response_model": f"ERROR: {e}",
                "answer_ground_truth": q.answer
            }
            # Save error file
            question_file = output_dir / f"q{question_num}_error.json"
            with question_file.open('w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)


    rprint(f"\n[bold green]Finished processing. All results saved to {str(output_dir)}/[/bold green]")


if __name__ == "__main__":
    main()
