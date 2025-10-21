import json
import os
import random
from datetime import datetime, UTC
from typing import Dict, Any, List

from load_llm import generate_with_openai

from run_base_prompt import evaluate_dataset

def generate_candidates_llm(base_system_prompt: str, sample_data: List[Dict[str, Any]],
                            model: str, num_candidates: int = 5) -> List[str]:

    # Build small sample of question-answer pairs
    sample_texts = []
    for item in random.sample(sample_data, 3):
        q = item.get("question", "")
        a = item.get("full_answer", "")
        q_prompt = f"{q}\nOutput final answer after ####. Example: #### ANSWER"
        sample_texts.append(f"Q: {q_prompt}\nA: {a}")
    sample_block = "\n\n".join(sample_texts)

    llm_instruction = f"""
        You are an expert prompt engineer. Given the following examples of question-answer pairs:\n\n
        {sample_block}\n\n
        Propose {num_candidates} distinct alternative system prompts that would help the model 
        answer these types of questions more accurately and return the answrrs in the right output format. 
        Each system prompt should be a single short paragraph. 
        Output a JSON array of strings and nothing else.
        Here is my best current system prompt: {base_system_prompt}
        """

    print("Asking LLM to generate candidate system prompts...")
    raw = generate_with_openai(prompt=llm_instruction, system_prompt="", model=model, max_tokens=1024)

    # Try to parse JSON directly
    candidates = []
    try:
        candidates = json.loads(raw)
        if isinstance(candidates, list) and all(isinstance(x, str) for x in candidates):
            return candidates[:num_candidates]
    except Exception:
        pass

    # Fallback: extract quoted lines or split by newlines and filter
    lines = [ln.strip(" \"'") for ln in raw.splitlines() if ln.strip()]
    # Keep lines that are a reasonable length
    filtered = [ln for ln in lines if 20 <= len(ln) <= 400]
    # Return unique candidates
    unique = []
    for ln in filtered:
        if ln not in unique:
            unique.append(ln)
    print("LLM candidate generation failed to return JSON; falling back to line-based parsing.")
    return unique[:num_candidates]


def optimize_system_prompt(data: List[Dict[str, Any]],
                           base_system_prompt: str = "",
                           model: str = "gpt-5-nano",
                           iterations: int = 3,
                           candidates_per_iter: int = 5) -> Dict[str, Any]:
    
    best_prompt = base_system_prompt or ""
    print(f"Starting optimization. base system prompt: '{best_prompt}'")
    # Get base prompt evaluation results
    best_eval = json.load(open("results/eval_base_prompt_gpt-5-nano.json"))
    best_accuracy = best_eval["results"]["accuracy"]
    print(f"Base prompt accuracy: {best_accuracy:.2f}%")

    # Record optimization history with key matrics
    history = [{
        "system_prompt": best_prompt,
        "accuracy": best_accuracy,
        "evaluation": best_eval
    }]

    # Loop through each iteration
    for it in range(1, iterations + 1):
        print(f"\n--- Optimization iteration {it}/{iterations} ---")
        # Generate candidates
        candidates = generate_candidates_llm(best_prompt, data, model=model, num_candidates=candidates_per_iter)
        if not candidates:
            print("LLM did not produce candidates! Proceed to the next iteration.")
            continue

        # Evaluate each candidate
        candidate_results = []
        for idx, cand in enumerate(candidates):
            print(f"\nEvaluating candidate {idx + 1}/{len(candidates)}:")
            print(f"Candidate system prompt: '{cand}'")
            eval_summary = evaluate_dataset(data, model=model, system_prompt=cand)
            acc = eval_summary["results"]["accuracy"]
            candidate_results.append({
                "system_prompt": cand,
                "accuracy": acc,
                "evaluation": eval_summary
            })
            print(f"Candidate accuracy: {acc:.2f}%")

        # Pick best candidate
        candidate_results.sort(key=lambda x: x["accuracy"], reverse=True)
        top = candidate_results[0]
        # Update best prompt if improved
        if top["accuracy"] > best_accuracy:
            print(f"New best prompt found with accuracy {top['accuracy']:.2f}% (improved from {best_accuracy:.2f}%).")
            best_prompt = top["system_prompt"]
            best_accuracy = top["accuracy"]
            best_eval = top["evaluation"]
        # Stop if no improvement
        else:
            print(f"No improvement found in iteration {it}. Best remains {best_accuracy:.2f}%.")
            break 

        history.append({
            "iteration": it,
            "top_candidate": top["system_prompt"],
            "top_accuracy": top["accuracy"],
            "all_candidates": candidate_results
        })

    # Save best prompt to a txt file
    output_dir = "results"
    timestamp = datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')
    best_prompt_path = os.path.join(output_dir, f"best_system_prompt_{model}.txt")
    with open(best_prompt_path, 'w', encoding='utf-8') as f:
        f.write(best_prompt)
    print(f"\nBest system prompt saved to: {best_prompt_path}")

    # Save optimization summary
    summary = {
        "metadata": {
            "timestamp": timestamp,
            "model": model,
            "user": os.getenv("USER", "unknown"),
            "iterations": iterations,
            "candidates_per_iter": candidates_per_iter
        },
        "best_prompt": best_prompt,
        "best_accuracy": best_accuracy,
        "history": history
    }
    summary_path = os.path.join(output_dir, f"eval_opto_summary_{model}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Optimization summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":

    # Load test data
    data = json.load(open("data/gsm8k.json"))

    optimize_system_prompt(
        data=data,
        base_system_prompt="",
        model="gpt-5-nano",
        iterations=10, # Maximum number of optimization iterations
        candidates_per_iter=3 # Number of candidate prompts to generate & evaluate per iteration
    )

