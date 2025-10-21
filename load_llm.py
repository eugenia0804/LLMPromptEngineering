import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

def generate_with_openai(prompt: str,
                         system_prompt: str = "",
                         model: str = "gpt-5-nano",
                         max_tokens: int = 150) -> str:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please ensure your .env file contains OPENAI_API_KEY=<your_key>")

    client = OpenAI(api_key=api_key)

    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # print(f"Sending request to OpenAI model '{model}' with prompt:\n{messages}\n")

    responses = client.responses.create(
        model=model,
        input=messages
    )

    # print("Received response from OpenAI: ", responses.output_text)

    return responses.output_text