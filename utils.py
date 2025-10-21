import re

def parse_answer(output: str) -> str:
    # Try to find answer after #### prefix
    for line in output.strip().splitlines()[::-1]:
        if line.strip().startswith('####'):
            return line.strip()[4:].strip()
    
    # Fallback: get last number
    numbers = re.findall(r"[-+]?\d*\.?\d+", output)
    return numbers[-1] if numbers else ""


def check_answer(parsed: str, expected: str) -> bool:
    try:
        # Try numeric comparison first
        return abs(float(parsed) - float(expected)) <= 1e-6
    except:
        # Fall back to string comparison
        return parsed.strip().lower() == expected.strip().lower()