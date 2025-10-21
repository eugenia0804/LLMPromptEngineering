import json
import os
import random
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
from load_llm import generate_with_openai
from utils import parse_answer, check_answer


def evaluate_prompt(
    question: str,
    expected: str,
    model: str = "gpt-5-nano",
    system_prompt: str = "",
    few_shot: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    
    # Build few-shot examples text (minimal, non-intrusive change)
    few_shot_text = ""
    if few_shot:
        examples = []
        for ex in few_shot:
            q = ex.get("question", "")
            a = "Reasoning: " + ex.get("full_answer", "")
            # Keep example format simple and compatible with the system prompt
            examples.append(f"Example Question:\n{q}\n {a}")
        few_shot_text = "\n\n".join(examples) + "\n\n"

    user_prompt = f"{few_shot_text}\nQuestion:\n{question}"

    # Get model response
    output = generate_with_openai(
        prompt=user_prompt,
        system_prompt=system_prompt,
        model=model
    )

    # Parse and check answer
    parsed = parse_answer(output)
    is_correct = check_answer(parsed, expected)

    return {
        "prompt": user_prompt,
        "output": output,
        "parsed": parsed,
        "expected": expected,
        "correct": is_correct
    }


def evaluate_dataset(
    data: List[Dict[str, Any]],
    model: str = "gpt-5-nano",
    num_few_shot: int = 2
) -> Dict[str, Any]:
    
    # Introduce best-practice system prompt
    system_prompt = (
        "You are an expert mathematics tutor who excels at solving grade-school "
        "math word problems. Follow these steps carefully:\n"
        "1. Read the question thoroughly and identify what is being asked.\n"
        "2. Think step-by-step through the reasoning process before deciding the answer.\n"
        "3. Show clear intermediate reasoning steps.\n"
        "4. Double-check your logic and arithmetic before finalizing.\n"
        "5. Provide your final numerical answer after the delimiter ####.\n\n"
        "Format your response exactly as:\n"
        "Reasoning: <your detailed reasoning here>\n"
        "#### <final numerical answer>"
    )

    results = []
    correct = 0
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    for item in data:

        # Randomly select a small number of few-shot examples
        pool = [x for x in data if x["id"] != item["id"]]
        few_shot_examples = random.sample(pool, num_few_shot)

        result = evaluate_prompt(
            question=item["question"],
            expected=item["final_answer"],
            model=model,
            system_prompt=system_prompt,
            few_shot=few_shot_examples
        )
        result["id"] = item["id"]
        results.append(result)

        if result["correct"]:
            correct += 1

        print(f"ID {item['id']}: got '{result['parsed']}', expected '{item['final_answer']}' -> {result['correct']}")

    pct = (correct / len(data) * 100) if data else 0
    print(f"\nImproved Prompt Accuracy: {correct}/{len(data)} = {pct:.1f}%")

    evaluation_summary = {
        "metadata": {
            "timestamp": timestamp,
            "model": model,
            "system_prompt": system_prompt,
            "user": os.getenv("USER", "unknown"),
            "total_examples": len(data),
            "few_shot_count": len(few_shot_examples),
        },
        "results": {
            "total": len(data),
            "correct": correct,
            "accuracy": pct,
            "evaluations": results,
        },
    }

    # Create output filename with timestamp and model
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"eval_few_shot_prompt_{model}.json"
    output_path = os.path.join(output_dir, filename)

    # Save results to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    return evaluation_summary


if __name__ == "__main__":
    # Load test data
    data = json.load(open("data/gsm8k.json"))
    
    # Run evaluation
    evaluate_dataset(
        data=data,
        model="gpt-5-nano",
        num_few_shot=2
    )