import os
import time
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,  StoppingCriteria, StoppingCriteriaList
from accelerate import Accelerator
from typing import List, Dict, Optional
import time
from datetime import datetime
from statistics import median
import torch.nn.functional as F

PRM_VARIANT_QWEN_MATH = "qwen2.5-math-prm-7b"
PRM_VARIANT_RLHFLOW_LLAMA = "rlhflow-llama3.1-8b-deepseek"
DEFAULT_QWEN_PRM_PATH = "./model_cache/models--Qwen--Qwen2.5-Math-PRM-7B/snapshots/0610740060112df12585d00a1c5f4624d2f59051"
DEFAULT_RLHFLOW_PRM_PATH = "./model_cache/models--RLHFlow--Llama3.1-8B-PRM-Deepseek-Data/snapshots/51100bc635eff6ad6a3e3fdab87703491461a88b"


def _normalize_prm_model_id(scorer_model: Optional[str]) -> tuple[str, Optional[str]]:
    """Map shorthand scorer identifiers to canonical defaults."""
    if not scorer_model:
        return DEFAULT_QWEN_PRM_PATH, PRM_VARIANT_QWEN_MATH
    normalized = scorer_model.strip().lower()
    if normalized in {"default", "qwen", "qwen-prm", "qwen2.5", "qwen2.5-math-prm-7b"}:
        return DEFAULT_QWEN_PRM_PATH, PRM_VARIANT_QWEN_MATH
    if normalized in {"rlhflow", "rlhflow-prm", "rlhflow/llama3.1-8b-prm-deepseek-data"}:
        return DEFAULT_RLHFLOW_PRM_PATH, PRM_VARIANT_RLHFLOW_LLAMA
    return scorer_model, None


def _is_local_model_path(model_id: str) -> bool:
    """Best-effort guess whether `model_id` resolves to a local directory."""
    expanded = os.path.expanduser(model_id)
    return (
        model_id.startswith("./")
        or model_id.startswith("/")
        or os.path.isdir(expanded)
    )


def _resolve_prm_loader(scorer_model: Optional[str]):
    """
    Decide which PRM variant and loader kwargs to use based on the provided model id/path.
    Returns (model_path, variant, model_cls, tokenizer_kwargs, model_kwargs).
    """
    model_path, variant_hint = _normalize_prm_model_id(scorer_model)
    normalized = model_path.lower()

    if variant_hint == PRM_VARIANT_RLHFLOW_LLAMA or any(key in normalized for key in ("rlhflow", "deepseek")):
        variant = PRM_VARIANT_RLHFLOW_LLAMA
        model_cls = AutoModelForCausalLM
        tokenizer_kwargs = {"device_map": "cuda"}
        model_kwargs = {"device_map": "cuda", "torch_dtype": "auto"}
    else:
        variant = PRM_VARIANT_QWEN_MATH
        model_cls = AutoModel
        tokenizer_kwargs = {"device_map": "cuda", "trust_remote_code": True}
        model_kwargs = {"device_map": "cuda", "torch_dtype": "auto", "trust_remote_code": True}

    local_only = _is_local_model_path(model_path)
    tokenizer_kwargs["local_files_only"] = local_only
    model_kwargs["local_files_only"] = local_only

    return model_path, variant, model_cls, tokenizer_kwargs, model_kwargs


def _detect_prm_variant(score_model) -> str:
    """Read the variant marker saved on the score model (defaults to Qwen PRM)."""
    return getattr(score_model, "_prm_variant", PRM_VARIANT_QWEN_MATH)

# ----------------------------------------------------------------------------
# 1. Setup and Initialization
# ----------------------------------------------------------------------------

def setup_models(model: str = "Qwen/Qwen3-1.7B", scorer_model: Optional[str] = None):
    """Initializes and returns the generator and PRM score models and tokenizers."""
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- Generator Model (e.g., a base Llama-3 or any capable LLM) ---
    # Using a smaller, fast model for demonstration. Replace with your preferred generator.
    generator_model_name = model
    if generator_model_name == "Qwen/Qwen3-1.7B":
        generator_model_path = "./model_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    elif generator_model_name == "Qwen/Qwen3-4B":
        generator_model_path = "./model_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    elif generator_model_name == "Qwen/Qwen3-8B":
        generator_model_path = "./model_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    elif generator_model_name == "llama-1b":
        generator_model_path = "./model_cache/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    elif generator_model_name == "llama-3b":
        generator_model_path = "./model_cache/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    elif generator_model_name == "llama-8b":
        generator_model_path = "./model_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    print(f"Loading generator model: {generator_model_path}")
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_path, device_map="cuda", local_files_only=True)
    generator_model = AutoModelForCausalLM.from_pretrained(
        generator_model_path, 
        local_files_only=True,
        torch_dtype="auto",
        device_map="cuda",  # Automatically map to available devices
        # quantization_config=quantization_config
    ).eval()

    print(f"Generator model loaded with {generator_model.config.num_hidden_layers} layers and {generator_model.config.hidden_size} hidden size.")

    # --- PRM Score Model ---
    score_model_path, prm_variant, score_model_cls, tokenizer_kwargs, model_kwargs = _resolve_prm_loader(scorer_model)
    print(f"Loading PRM score model ({prm_variant}): {score_model_path}")
    score_tokenizer = AutoTokenizer.from_pretrained(score_model_path, **tokenizer_kwargs)
    score_model = score_model_cls.from_pretrained(
        score_model_path,
        **model_kwargs,
    ).eval()
    score_model._prm_variant = prm_variant
    score_model._prm_model_path = score_model_path

    return (
        generator_model, generator_tokenizer,
        score_model, score_tokenizer,
        device
    )

# ----------------------------------------------------------------------------
# 2. Scoring Function (Adapted from your logic)
# ----------------------------------------------------------------------------

def _format_response_for_prm(raw_response: str) -> str:
    """Convert raw assistant text into the <extra_0>-delimited format expected by the PRM."""
    steps = [step.strip() for step in raw_response.split("\n\n") if step.strip()]
    if not steps:
        return ""
    return "<extra_0>".join(steps) + "<extra_0>"


def _make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor) -> list[list[float]]:
    """Replicates the step reward extraction logic provided in the Qwen PRM example."""
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1).to(probabilities.dtype)

    all_scores_res: list[list[float]] = []
    for sample in probabilities:
        non_zero = sample[sample != 0]
        if non_zero.numel() < 2:
            all_scores_res.append([])
            continue
        # Each valid step yields two labels; guard against odd counts by truncation.
        usable = non_zero[: (non_zero.numel() // 2) * 2]
        positive_probs = usable.view(-1, 2)[:, 1]
        all_scores_res.append(positive_probs.cpu().tolist())
    return all_scores_res


def _score_branches_qwen_prm(
    branch_texts: list[str],
    score_model,
    score_tokenizer,
    device: torch.device,
    initial_prompt_text: str,
    base_messages: List[Dict[str, str]],
) -> list[float]:
    """Score branches with the Qwen2.5-Math PRM (step-wise rewards)."""
    try:
        step_sep_id = score_tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
    except (IndexError, ValueError) as exc:
        raise ValueError("Could not encode '<extra_0>' token with the score_tokenizer.") from exc

    scores: list[float] = []

    for text in branch_texts:
        assistant_response = text[len(initial_prompt_text):] if text.startswith(initial_prompt_text) else text
        formatted_response = _format_response_for_prm(assistant_response)
        if not formatted_response:
            scores.append(0.0)
            continue

        conversation = [dict(msg) for msg in base_messages]
        conversation.append({"role": "assistant", "content": formatted_response})

        input_ids = score_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = score_model(input_ids=input_ids)

        logits = outputs[0]

        token_masks = (input_ids == step_sep_id)
        step_rewards = _make_step_rewards(logits, token_masks)
        if step_rewards and step_rewards[0]:
            scores.append(step_rewards[0][-1])
        else:
            scores.append(0.0)

    return scores


def _score_branches_rlhflow_prm(
    branch_texts: list[str],
    score_model,
    score_tokenizer,
    device: torch.device,
) -> list[float]:
    """Score branches with the RLHFlow Llama-3.1 PRM using '+' likelihood."""
    if not branch_texts:
        return []
    try:
        plus_token_id = score_tokenizer.encode('+', add_special_tokens=False)[0]
    except (IndexError, ValueError) as exc:
        raise ValueError("Could not encode '+' token with the score_tokenizer.") from exc

    scores: list[float] = []
    for text in branch_texts:
        conversation = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "+"},
        ]
        input_ids = score_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = score_model(input_ids).logits

        last_logit = logits[0, -1, :]
        probabilities = torch.softmax(last_logit, dim=-1)
        scores.append(probabilities[plus_token_id].item())
    return scores


def score_branches_prm(
    branch_texts: list[str],
    score_model,
    score_tokenizer,
    device: torch.device,
    initial_prompt_text: str,
    base_messages: List[Dict[str, str]],
) -> list[float]:
    """
    Scores candidate branches with the configured PRM model, returning a scalar reward per branch.
    """
    prm_variant = _detect_prm_variant(score_model)
    if prm_variant == PRM_VARIANT_RLHFLOW_LLAMA:
        scores = _score_branches_rlhflow_prm(branch_texts, score_model, score_tokenizer, device)
    else:
        scores = _score_branches_qwen_prm(
            branch_texts,
            score_model,
            score_tokenizer,
            device,
            initial_prompt_text,
            base_messages,
        )

    print(f"PRM ({prm_variant}) Scores: {scores}")
    return scores


class StopOnSequences(StoppingCriteria):
    """
    Custom stopping criteria to stop generation when any of a list of
    token sequences is found in the *newly generated* text.
    """
    def __init__(self, stop_sequences: List[List[int]], device: torch.device, prompt_len_to_ignore: int):
        super().__init__()
        self.stop_sequences = [torch.tensor(seq, dtype=torch.long, device=device) for seq in stop_sequences]
        self.prompt_len_to_ignore = prompt_len_to_ignore

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check only the newly generated tokens
        generated_ids = input_ids[0, self.prompt_len_to_ignore:]
        
        # If no new tokens are generated, don't stop
        if generated_ids.numel() == 0:
            return False
            
        for stop_seq in self.stop_sequences:
            # Check if the generated sequence ends with the stop sequence
            if len(generated_ids) >= len(stop_seq):
                if torch.equal(generated_ids[-len(stop_seq):], stop_seq):
                    return True # Stop generation
        return False # Continue generation

# ----------------------------------------------------------------------------
# 3. Main Generation Loop (Corrected and with Debug Prints)
# ----------------------------------------------------------------------------

def reasoning_beam_search(
    messages: List[Dict[str, str]],
    generator_model,
    generator_tokenizer,
    score_model,
    score_tokenizer,
    device: torch.device,
    num_main_beams: int = 3,
    num_expansions_per_beam: int = 5,
    num_reasoning_steps: int = 5,
    max_step_tokens: int = 100
):
    """
    Performs per-reasoning-step beam search, using a PRM for scoring.
    """
    
    def _now_ts():
        return datetime.utcnow().isoformat() + "Z"
    run_start = time.perf_counter()
    run_started_at = _now_ts()

    # --- Define End-of-Generation and Padding Tokens ---
    if generator_tokenizer.pad_token_id is None:
        generator_tokenizer.pad_token = generator_tokenizer.eos_token
        print(f"Warning: pad_token_id not set. Using eos_token: {generator_tokenizer.eos_token}")

    final_eos_token_id = generator_tokenizer.eos_token_id
    # Handle tokenizers that might have multiple eos_tokens (like Llama3)
    # We choose one canonical one to be the "final" one.
    if isinstance(final_eos_token_id, list):
        final_eos_token_id = final_eos_token_id[0]
        
    print(f"Using final End-of-Generation token ID: {final_eos_token_id} ('{generator_tokenizer.decode(final_eos_token_id)}')")

    # --- Define Stop Logic for Intermediate Steps ---
    step_stop_strings = ["\n\n", ".\n", ":\n", "\n", ".\n\n", ":\n\n"]
    step_stop_token_id_seqs = [
        generator_tokenizer.encode(s, add_special_tokens=False) for s in step_stop_strings
    ]
    print(f"Intermediate step stop sequences (token IDs): {step_stop_token_id_seqs}")
    
    # --- Initialize telemetry (no env) ---
    trace = {
        "started_at": run_started_at,
        "config": {
            "num_main_beams": num_main_beams,
            "num_expansions_per_beam": num_expansions_per_beam,
            "num_reasoning_steps": num_reasoning_steps,
            "max_step_tokens": max_step_tokens
        }
    }


    # --- Define Tokens to Suppress ---
    # We suppress special tokens during intermediate generation to avoid confusion,
    # but we MUST ALLOW the final_eos_token_id to be generated.
    suppress_tokens = []
    special_ids = generator_tokenizer.all_special_ids
    for token_id in special_ids:
        if token_id != final_eos_token_id:
            suppress_tokens.append(token_id)
    print(f"Suppressing the following token IDs during intermediate steps: {suppress_tokens}")
    
    # --- Get the token ID for the definitive end token ---
    if "qwen" in generator_model.config._name_or_path.lower():
        # --- Initialize beam lists ---
        initial_input_text = generator_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    elif "llama" in generator_model.config._name_or_path.lower():
        # --- Initialize beam lists ---
        initial_input_text = generator_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        raise ValueError("Unsupported model identification.")

    active_beams = [{"text": initial_input_text, "step_score": 1.0, "new_text": "[Initial Prompt]"}]
    completed_beams = []

    for step in range(num_reasoning_steps):
        print(f"\n{'='*20} Reasoning Step {step+1}/{num_reasoning_steps} {'='*20}")

        if not active_beams:
            print("Termination: No active beams left to expand.")
            break

        all_candidate_branches = []
        # --- A. Expansion Phase ---
        print(f"Expanding {len(active_beams)} active beams...")
        for i, beam in enumerate(active_beams):
            print(f"  - Expanding Beam #{i+1}...")
            input_ids = generator_tokenizer(beam["text"], return_tensors="pt").input_ids.to(device)
            prompt_len = input_ids.shape[1]
            parent_text = beam["text"]
            
            # print(f"    - Prompt Text: \"{parent_text[-100:]}...\"")
            # print(f"    - Prompt Length: {prompt_len} tokens")

            # --- ### CHANGE: Create stopper FOR EACH beam expansion ### ---
            step_stopper = StopOnSequences(step_stop_token_id_seqs, device, prompt_len_to_ignore=prompt_len)
            stopping_criteria = StoppingCriteriaList([step_stopper])

            generated_sequences = generator_model.generate(
                input_ids,
                num_return_sequences=num_expansions_per_beam,
                max_new_tokens=max_step_tokens,
                min_new_tokens=5,
                pad_token_id=generator_tokenizer.pad_token_id,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                suppress_tokens=suppress_tokens
            )

            for seq in generated_sequences:
                ### --- THE FIX IS HERE --- ###
                
                # 1. Get the tensor of ONLY the new tokens by slicing the token tensor.
                new_token_ids = seq[prompt_len:]
                
                # 2. Decode ONLY the new tokens. This is guaranteed to be correct.
                #    Use skip_special_tokens=False first to see everything, then switch to True.
                new_text = generator_tokenizer.decode(new_token_ids, skip_special_tokens=False)
                # print(f"    - Generated New Text: \"{new_text.strip()}\"")

                last_token_id = seq[-1].item()
                # print(f"    - Last Token ID: {last_token_id} ('{generator_tokenizer.decode(last_token_id)}')")
                # 3. Construct the full text by combining the original text and the new part.
                full_text = beam["text"] + new_text
                
                # The rest of the logic can now use these correct variables
                all_candidate_branches.append({
                    "text": full_text, 
                    "new_text": new_text,
                    "parent_beam": beam, 
                    "last_token_id": seq[-1].item()
                })


        if not all_candidate_branches:
            print("Warning: No new branches generated. Halting.")
            break
        if all(not c['new_text'].strip() for c in all_candidate_branches):
            print("Termination: Generation has stagnated (no new text produced).")
            break

        # --- B. Scoring Phase (INTEGRATION POINT) ---
        print(f"Scoring {len(all_candidate_branches)} candidate branches with PRM...")
        branch_texts_to_score = [b["text"] for b in all_candidate_branches]

        # <<< CHANGE: Use the new PRM scoring function instead of a placeholder >>>
        branch_scores = score_branches_prm(
            branch_texts_to_score,
            score_model,
            score_tokenizer,
            device,
            initial_input_text,
            messages,
        )
        # <<< END CHANGE >>>

        for candidate, score in zip(all_candidate_branches, branch_scores):
            # In PRM scoring, we care about the current step's quality, so we don't
            # necessarily need to accumulate. Using the raw score is often better.
            # If you want to reward length, you can use accumulation.
            candidate["step_score"] = score

        # --- C. Partitioning and Selection Phase ---
        print("Partitioning into completed and new active beams...")
        all_candidates_for_ranking = all_candidate_branches + completed_beams
        sorted_candidates = sorted(all_candidates_for_ranking, key=lambda x: x["step_score"], reverse=True)

        new_active_beams, new_completed_beams = [], []
        for candidate in sorted_candidates:
            is_complete = candidate.get("last_token_id") == final_eos_token_id
            if is_complete:
                new_completed_beams.append(candidate)
            elif len(new_active_beams) < num_main_beams:
                new_active_beams.append(candidate)

        active_beams = new_active_beams
        completed_beams += new_completed_beams
        
        # --- Print Status ---
        print(f"Status update: {len(active_beams)} active beams, {len(completed_beams)} completed beams.")
        if active_beams:
            print("  --- Top Active Beams ---")
            for beam in active_beams:
                print(f"    Score: {beam['step_score']:.4f} | New Text: \"{beam['new_text'].strip()}\"")
        if completed_beams:
            print("  --- Top Completed Beams ---")
            for beam in completed_beams:
                print(f"    Score: {beam['step_score']:.4f} | New Text: \"{beam['new_text'].strip()}\"")

        if len(completed_beams) + len(active_beams) >= num_main_beams*num_expansions_per_beam:
            print("Termination: No beams left to continue.")
            break

    # --- Final Selection ---
    print("\n--- Final Selection ---")
    final_candidates = completed_beams
    if not final_candidates:
        run_end = time.perf_counter()
        trace["ended_at"] = _now_ts()
        trace["total_wall_time_s"] = run_end - run_start
        result = {"text": initial_input_text, "step_score": -1, "new_text": "No generation.", "trace": trace}
        return result

    best_beam = sorted(final_candidates, key=lambda x: x["step_score"], reverse=True)[0]

    run_end = time.perf_counter()
    trace["ended_at"] = _now_ts()
    trace["total_wall_time_s"] = run_end - run_start

    best_beam["trace"] = trace

    return best_beam
