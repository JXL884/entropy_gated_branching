import time
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List, Dict
from datetime import datetime
from statistics import median

from beam_search_vers import score_branches_prm, setup_models


def calculate_entropy_from_logits(logits: torch.Tensor) -> float:
    """Calculates the entropy of a single distribution of logits."""
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy().item()

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
# 3. Main Generation Loop
# ----------------------------------------------------------------------------

def reasoning_confidence_beam_search(
    messages: List[Dict[str, str]],
    generator_model,
    generator_tokenizer,
    score_model,
    score_tokenizer,
    device: torch.device,
    num_main_beams: int = 3,
    num_expansions_per_beam: int = 5,
    num_reasoning_steps: int = 5,
    max_step_tokens: int = 2048,
    entropy_threshold: float = 2.0,
):
    """
    Performs reasoning search that FORCES a wide expansion on the first step,
    then uses confidence-based expansion for all subsequent steps.
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
            "max_step_tokens": max_step_tokens,
            "entropy_threshold": entropy_threshold,
        },
        "steps": [],
        "high_entropy_events": [],  # flat list across all steps
        "counters": {
            "total_candidate_branches": 0,
            "total_expansions": 0,
            "num_high_entropy_triggers": 0,
            "approx_tokens_generated": 0
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

        step_rec = {
            "step_index": step,
            "beam_count_in": len(active_beams),
            "started_at": _now_ts(),
            "timings": {},
            "expansion": {"mode": "forced" if step == 0 else "confidence",
                          "num_expanded_total": 0, "per_beam": []},
            "scoring": {},
            "selection": {}
        }
        t_step0 = time.perf_counter()

        all_candidate_branches = []

        # --- A. Expansion Phase (Hybrid Logic) ---
        t_gen0 = time.perf_counter()
        for i, beam in enumerate(active_beams):
            beam_info = {
                "beam_index": i,
                "parent_len_chars": len(beam["text"]),
                "high_entropy_trigger": None,
                "num_expanded_from_beam": 0
            }
            print(f"  --- Processing Beam {i+1}/{len(active_beams)} ---")
            parent_text = beam["text"]
            input_ids = generator_tokenizer(parent_text, return_tensors="pt").input_ids.to(device)
            original_length = input_ids.shape[1]

            # --- ### CHANGE: Create stopper FOR EACH beam expansion ### ---
            step_stopper = StopOnSequences(step_stop_token_id_seqs, device, prompt_len_to_ignore=original_length+1)
            stopping_criteria = StoppingCriteriaList([step_stopper])

            if step == 0:
                # --- STEP 1: FORCE a wide, sampled expansion ---
                print("  Step 1: Forcing initial expansion to create diverse starting points.")
                t_local0 = time.perf_counter()
                expanded_sequences = generator_model.generate(
                    input_ids,
                    max_new_tokens=max_step_tokens,
                    min_new_tokens=5,
                    num_return_sequences=num_expansions_per_beam,
                    do_sample=True,
                    pad_token_id=generator_tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria,
                    suppress_tokens=suppress_tokens
                )
                t_local1 = time.perf_counter()

                for seq in expanded_sequences:
                    new_token_ids = seq[original_length:]
                    last_token_id = seq[-1].item()
                    new_text = generator_tokenizer.decode(new_token_ids, skip_special_tokens=False)
                    full_text = beam["text"] + new_text

                    # full_text = generator_tokenizer.decode(seq, skip_special_tokens=True)
                    # new_text = full_text[len(parent_text):]
                    all_candidate_branches.append({
                        "text": full_text, "new_text": new_text,
                        "parent_beam": beam, "last_token_id": seq[-1].item()
                    })
                    trace["counters"]["approx_tokens_generated"] += max(int(seq.shape[-1] - original_length), 0)

                n_expanded = len(expanded_sequences)
                step_rec["expansion"]["num_expanded_total"] += n_expanded
                beam_info["num_expanded_from_beam"] = n_expanded
                beam_info["gen_wall_time_s"] = t_local1 - t_local0

            else:
                # --- STEPS 2+: Confidence-based expansion ---
                t_conf0 = time.perf_counter()
                confident_outputs = generator_model.generate(
                    input_ids,
                    max_new_tokens=max_step_tokens,
                    min_new_tokens=5,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=generator_tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria,
                    suppress_tokens=suppress_tokens
                )
                t_conf1 = time.perf_counter()

                # Inspect for high entropy
                expansion_triggered = False
                entropies = []
                for token_idx, token_scores in enumerate(confident_outputs.scores):
                    H = calculate_entropy_from_logits(token_scores[0])  # your function
                    entropies.append(float(H))
                    if H > entropy_threshold:
                        t_trigger = time.perf_counter()
                        print(f"  -------------- High entropy ({H:.2f} > {entropy_threshold}) at token {token_idx+1}. Rolling back. ---------------")

                        rollback_point_ids = confident_outputs.sequences[:, :original_length + token_idx]
                        t_rb0 = time.perf_counter()
                        expanded_sequences = generator_model.generate(
                            rollback_point_ids,
                            max_new_tokens=max_step_tokens - token_idx,
                            min_new_tokens=5,
                            num_return_sequences=num_expansions_per_beam,
                            do_sample=True,
                            pad_token_id=generator_tokenizer.pad_token_id,
                            stopping_criteria=stopping_criteria,
                            suppress_tokens=suppress_tokens
                        )
                        t_rb1 = time.perf_counter()

                        for seq in expanded_sequences:
                            new_token_ids = seq[original_length + token_idx:]
                            last_token_id = seq[-1].item()
                            new_text = generator_tokenizer.decode(new_token_ids, skip_special_tokens=False)
                            full_text = beam["text"] + new_text
                            all_candidate_branches.append({
                                "text": full_text, "new_text": new_text,
                                "parent_beam": beam, "last_token_id": seq[-1].item()
                            })
                            trace["counters"]["approx_tokens_generated"] += max(int(seq.shape[-1] - original_length - token_idx), 0)

                        n_expanded = len(expanded_sequences)
                        step_rec["expansion"]["num_expanded_total"] += n_expanded
                        beam_info["num_expanded_from_beam"] = n_expanded
                        beam_info["high_entropy_trigger"] = {
                            "token_idx": int(token_idx),
                            "entropy": float(H),
                            "time_since_run_start_s": t_trigger - run_start,
                            "time_since_step_start_s": t_trigger - t_step0
                        }
                        trace["high_entropy_events"].append({
                            "step_index": step,
                            "beam_index": i,
                            "token_idx": int(token_idx),
                            "entropy": float(H),
                            "time_since_run_start_s": t_trigger - run_start,
                            "time_since_step_start_s": t_trigger - t_step0
                        })
                        trace["counters"]["num_high_entropy_triggers"] += 1
                        beam_info["confident_gen_wall_time_s"] = t_conf1 - t_conf0
                        beam_info["rollback_expand_wall_time_s"] = t_rb1 - t_rb0
                        expansion_triggered = True
                        break

                if not expansion_triggered:
                    print("  Confident path accepted. No high entropy detected.")
                    seq = confident_outputs.sequences[0]
                    new_token_ids = seq[original_length:]
                    last_token_id = seq[-1].item()
                    new_text = generator_tokenizer.decode(new_token_ids, skip_special_tokens=False)
                    full_text = beam["text"] + new_text
                    all_candidate_branches.append({
                        "text": full_text, "new_text": new_text,
                        "parent_beam": beam, "last_token_id": seq[-1].item()
                    })
                    trace["counters"]["approx_tokens_generated"] += max(int(seq.shape[-1] - original_length), 0)
                    beam_info["confident_gen_wall_time_s"] = t_conf1 - t_conf0

            step_rec["expansion"]["per_beam"].append(beam_info)

        t_gen1 = time.perf_counter()
        step_rec["timings"]["generation_wall_time_s"] = t_gen1 - t_gen0
        trace["counters"]["total_expansions"] += step_rec["expansion"]["num_expanded_total"]

        # --- B. Scoring & Selection Phases ---
        if not all_candidate_branches:
            print("Warning: No new branches generated. Halting.")
            break
        if all(not c['new_text'].strip() for c in all_candidate_branches):
            print("Termination: Generation has stagnated (no new text produced).")
            break

        trace["counters"]["total_candidate_branches"] += len(all_candidate_branches)

        print(f"\nScoring {len(all_candidate_branches)} total candidate branches with PRM...")
        t_score0 = time.perf_counter()
        branch_texts_to_score = [b["text"] for b in all_candidate_branches]
        branch_scores = score_branches_prm(
            branch_texts_to_score,
            score_model,
            score_tokenizer,
            device,
            initial_input_text,
            messages,
        )
        t_score1 = time.perf_counter()

        for candidate, score in zip(all_candidate_branches, branch_scores):
            candidate["step_score"] = float(score)

        step_rec["scoring"] = {
            "num_scored": len(all_candidate_branches),
            "wall_time_s": t_score1 - t_score0,
            "score_max": max(branch_scores),
            "score_median": median(branch_scores) if len(branch_scores) > 1 else float(branch_scores[0]),
        }

        print("Partitioning and selecting best beams...")
        t_sel0 = time.perf_counter()
        new_completed_beams = list(completed_beams)
        sorted_candidates = sorted(all_candidate_branches, key=lambda x: x["step_score"], reverse=True)

        new_active_beams, new_completed_beams = [], []
        for candidate in sorted_candidates:
            is_complete = candidate.get("last_token_id") == final_eos_token_id
            if is_complete:
                new_completed_beams.append(candidate)
            elif len(new_active_beams) < num_main_beams:
                new_active_beams.append(candidate)

        active_beams = new_active_beams
        completed_beams += new_completed_beams

        t_sel1 = time.perf_counter()

        step_rec["selection"] = {
            "active_after": len(active_beams),
            "completed_after": len(completed_beams),
            "wall_time_s": t_sel1 - t_sel0
        }

        print(f"Status update: {len(active_beams)} active beams, {len(completed_beams)} completed beams.")
        if active_beams:
            print("  --- Top Active Beams ---")
            for i, beam in enumerate(active_beams):
                print(f"    {i+1}. Score: {beam['step_score']:.4f} | New Text: \"{beam['new_text'].strip()}\"")
        if completed_beams:
            print("  --- Top Completed Beams ---")
            for i, beam in enumerate(completed_beams):
                print(f"    {i+1}. Score: {beam['step_score']:.4f} | Final Text: \"{beam['new_text'].strip()}\"")
        if len(completed_beams) >= num_main_beams:
            print(f"\nTermination: All {num_main_beams} beam slots are filled with completed sequences.")
            break

        step_rec["ended_at"] = _now_ts()
        t_step1 = time.perf_counter()
        step_rec["timings"]["step_wall_time_s"] = t_step1 - t_step0
        trace["steps"].append(step_rec)

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

    # Attach a small summary about the winner
    try:
        best_seq_len_tokens = len(generator_tokenizer(best_beam["text"]).input_ids)
    except Exception:
        best_seq_len_tokens = None

    trace["best_beam"] = {
        "step_score": float(best_beam.get("step_score", -1)),
        "is_complete": bool(best_beam.get("last_token_id") == final_eos_token_id),
        "length_chars": len(best_beam.get("text", "")),
        "length_tokens": best_seq_len_tokens
    }

    best_beam["trace"] = trace

    return best_beam
