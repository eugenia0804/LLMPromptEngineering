import json
import os
from datetime import datetime, UTC
from typing import Dict, Any, List
from load_llm import generate_with_openai
from utils import parse_answer, check_answer

def evaluate_prompt(question: str, expected: str, 
                   model: str = "gpt-5-nano",
                   system_prompt: str = "") -> Dict[str, Any]:
    
    # Add instruction to format answer
    prompt = f"{question}\nOutput final answer after ####. Example: #### ANSWER"
    
    # Get model response
    output = generate_with_openai(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=150
    )
    
    # Parse and check answer
    parsed = parse_answer(output)
    is_correct = check_answer(parsed, expected)
    
    return {
        "prompt": prompt,
        "output": output, 
        "parsed": parsed,
        "expected": expected,
        "correct": is_correct
    }

def evaluate_dataset(data: List[Dict[str, Any]], 
                    model: str = "gpt-5-nano",
                    system_prompt: str = "",
                    save_res: bool = False) -> Dict[str, Any]:
    """Evaluate all prompts in dataset."""
    results = []
    correct = 0

    timestamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
    
    for item in data:
        result = evaluate_prompt(
            question=item["question"], 
            expected=item["final_answer"],
            model=model,
            system_prompt=system_prompt
        )
        result["id"] = item["id"]
        results.append(result)
        
        if result["correct"]:
            correct += 1
            
        print(f"ID {item['id']}: got '{result['parsed']}', expected '{item['final_answer']}' -> {result['correct']}")
    
    pct = (correct / len(data) * 100) if data else 0
    print(f"\nAccuracy: {correct}/{len(data)} = {pct:.1f}%")
    
    evaluation_summary = {
        "metadata": {
            "timestamp": timestamp,
            "model": model,
            "system_prompt": system_prompt,
            "user": os.getenv("USER", "unknown"),
            "total_examples": len(data)
        },
        "results": {
            "total": len(data),
            "correct": correct,
            "accuracy": pct,
            "evaluations": results
        }
    }
    
    if save_res:
        # Create output filename with timestamp and model
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"eval_base_prompt_{model}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Save results to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
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
        save_res=True
    )