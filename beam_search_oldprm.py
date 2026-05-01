# ----------------------------------------------------------------------------
# 1. Setup and Initialization
# ----------------------------------------------------------------------------

def setup_models(model: str = "Qwen/Qwen3-1.7B"):
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
        generator_model_path = "Qwen/Qwen3-1.7B"
    elif generator_model_name == "Qwen/Qwen3-4B":
        generator_model_path = "Qwen/Qwen3-4B"
    elif generator_model_name == "Qwen/Qwen3-8B":
        generator_model_path = "Qwen/Qwen3-8B"
    elif generator_model_name == "llama-1b":
        generator_model_path = "meta-llama/Llama-3.2-1B-Instruct"
    elif generator_model_name == "llama-3b":
        generator_model_path = "meta-llama/Llama-3.2-3B-Instruct"
    elif generator_model_name == "llama-8b":
        generator_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading generator model: {generator_model_path}")
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_path, device_map="cuda", token="REDACTED_HF_TOKEN" )
    generator_model = AutoModelForCausalLM.from_pretrained(
        generator_model_path, 
        torch_dtype="auto",
        token="REDACTED_HF_TOKEN",
        device_map="cuda",  # Automatically map to available devices
        # quantization_config=quantization_config
    )
    print(f"Generator model loaded with {generator_model.config.num_hidden_layers} layers and {generator_model.config.hidden_size} hidden size.")
    # --- PRM Score Model ---
    # score_model_name = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    score_model_path = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    print(f"Loading PRM score model: {score_model_path}")
    # The tokenizer for the PRM might be different, but often it's the same base
    score_tokenizer = AutoTokenizer.from_pretrained(score_model_path, device_map="cuda",  token="REDACTED_HF_TOKEN")
    score_model = AutoModelForCausalLM.from_pretrained(
        score_model_path,
        torch_dtype="auto",
        device_map="cuda",  # Automatically map to available devices
        quantization_config=quantization_config,
        token="REDACTED_HF_TOKEN"
    )

    return (
        generator_model, generator_tokenizer,
        score_model, score_tokenizer,
        device
    )
# ----------------------------------------------------------------------------
# 2. Scoring Function (Adapted from your logic)
# ----------------------------------------------------------------------------

def score_branches_prm(
    branch_texts: list[str],
    score_model, # : AutoModelForCausalLM,
    score_tokenizer, # : AutoTokenizer,
    device: torch.device
) -> list[float]:
    """
    Scores a list of candidate branch texts using a Preference Ranking Model (PRM).

    The PRM is prompted with each candidate and asked to predict a '+' token,
    indicating it's a good continuation. The probability of the '+' token is
    used as the score for that branch.

    Args:
        branch_texts: A list of full text strings for each candidate branch.
        score_model: The loaded preference model.
        score_tokenizer: The tokenizer for the preference model.
        device: The torch device to run the model on.

    Returns:
        A list of float scores, one for each branch text.
    """
    scores = []
    
    # Get the token ID for the positive evaluation token ('+').
    # We only need the raw token ID, not the special start/end tokens.
    try:
        plus_token_id = score_tokenizer.encode('+', add_special_tokens=False)[0]
    except IndexError:
        raise ValueError("Could not encode '+' token with the score_tokenizer.")

    for text in branch_texts:
        # 1. Format the input for the PRM model.
        # The user/prompt part is the entire candidate text.
        # The assistant part is just '+', for which we want to get the probability.
        conversation = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "+"}
        ]
        
        # 2. Tokenize the input.
        # We don't need a batch here, we process one by one.
        input_ids = score_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        # 3. Get the logits from the model.
        with torch.no_grad():
            logits = score_model(input_ids).logits

        # 4. Find the score for the '+' token.
        # The logits for the *next* token are at the -2 position of the sequence.
        # The last position (-1) is the token ID for '+', so we want the model's
        # prediction leading up to it.
        # Your original code used -3; -2 is often more robust, but this can be
        # sensitive to the exact chat template. Adjust if needed.
        last_logit = logits[0, -1, :] 
        
        # 5. Calculate the probability of the '+' token.
        probabilities = torch.softmax(last_logit, dim=-1)
        score = probabilities[plus_token_id].item()
        
        scores.append(score)
        
    return scores

# ----------------------------------------------------------------------------
# 3. Main Generation Loop
# ----------------------------------------------------------------------------

def reasoning_beam_search(
    messages: List[str],
    generator_model, # : AutoModelForCausalLM,
    generator_tokenizer, # : AutoTokenizer,
    score_model, # : AutoModelForCausalLM,
    score_tokenizer, # : AutoTokenizer,
    device: torch.device,
    num_main_beams: int = 3,
    num_expansions_per_beam: int = 5,
    num_reasoning_steps: int = 100,
    max_step_tokens: int = 2048
):
    """
    Performs per-reasoning-step beam search, using a PRM for scoring.
    """
    run_start = time.perf_counter()
    # --- Get the token ID for the definitive end token ---
    if "qwen" in generator_model.config._name_or_path.lower():
        end_token_id = generator_tokenizer.convert_tokens_to_ids("<|im_end|>")
    elif "llama" in generator_model.config._name_or_path.lower():
        end_token_id = 128001
    else:
        raise ValueError("Unsupported end token identification.")
    
    # --- Define stop tokens for ending a *step* ---
    step_stop_strings = ["\n\n", ".\n\n", ":\n\n"]
    step_stop_token_ids = list(set(
        tok_id for s in step_stop_strings 
        for tok_id in generator_tokenizer.encode(s, add_special_tokens=False)
    ))
    if generator_tokenizer.eos_token_id is not None:
        step_stop_token_ids.append(generator_tokenizer.eos_token_id)
    step_stop_token_ids.append(end_token_id)
    step_stop_token_ids = list(set(step_stop_token_ids))
    print(f"Step will stop at token IDs: {step_stop_token_ids}")

    # --- Initialize beam lists ---
    initial_input_text = generator_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    initial_input_text = initial_input_text + "\n\n"

    active_beams = [{"text": initial_input_text, "step_score": 0.0, "new_text": "[Initial Prompt]"}]
    completed_beams = []

    for step in range(num_reasoning_steps):
        print(f"\n{'='*20} Reasoning Step {step+1}/{num_reasoning_steps} {'='*20}")

        if not active_beams:
            print("Termination: No active beams left to expand.")
            break

        all_candidate_branches = []
        # --- A. Expansion Phase ---
        print(f"Expanding {len(active_beams)} active beams...")
        for beam in active_beams:
            input_ids = generator_tokenizer(beam["text"], return_tensors="pt").input_ids.to(device)
            parent_text = beam["text"]
            
            generated_sequences = generator_model.generate(
                input_ids, num_return_sequences=num_expansions_per_beam,
                max_new_tokens=max_step_tokens, 
                eos_token_id=step_stop_token_ids,
                min_new_tokens=5,  # <-- ADD THIS LINE
                pad_token_id=generator_tokenizer.pad_token_id,
                do_sample=True,
            )

            for seq in generated_sequences:
                full_text = generator_tokenizer.decode(seq, skip_special_tokens=True)
                new_text = full_text[len(parent_text):]
                all_candidate_branches.append({
                    "text": full_text, "new_text": new_text,
                    "parent_beam": beam, "last_token_id": seq[-1].item()
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
            branch_texts_to_score, score_model, score_tokenizer, device
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
            is_complete = candidate.get("last_token_id") == end_token_id
            if is_complete:
                    new_completed_beams.append(candidate)
            else:
                if len(new_active_beams) < num_main_beams:
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
        return {"text": initial_input_text, "step_score": -1}

    best_beam = sorted(final_candidates, key=lambda x: x["step_score"], reverse=True)[0]

    run_end = time.perf_counter()
    best_beam["total_time"] = run_end - run_start

    return best_beam