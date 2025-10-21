## LLMPromptEngineering

### Overview

This project is set up for experimenting with and evaluating various prompt engineering techniques to improve the performance of Large Language Models (LLMs) on a specific task. The goal is to measure how different prompting strategies affect accuracy on the GSM8K dataset, which consists of grade-school math word problems.

The codebase systematically compares 4 distinct approaches:

- Base Prompt: A simple, zero-shot prompt with only formatting instructions.
- Improved Prompt: A "best-practice" zero-shot prompt that includes a persona, step-by-step reasoning instructions, and output formatting.
- Few-Shot Prompt: Augments the improved prompt with randomly selected in-context examples.
- OPRO-like Optimization: An iterative, automated process where an LLM generates and evaluates new system prompts to find an optimal one.

All experiment results can be found in the `results/` folder.

### How to Run the Codebase

1. Create a `.env` file in the root directory containing your OpenAI API key: 
    ```
    OPENAI_API_KEY=sk-your-open-ai-api-key
    ```
2. Build the Docker container: 
   ```
   docker build -t llm-runner .
   ```
3. Run the container to prepare the data. This executes the data_processing.py script, which downloads the GSM8K dataset, samples 250 questions, and saves them to data/gsm8k.json: 
   ```
   docker run --env-file .env llm-runner
   ```
4. Run each experiment. The results for each run will be saved to the results directory (which is created automatically):
    ```
    # 1. Run the baseline evaluation
    docker run --env-file .env llm-runner python run_base_prompt.py

    # 2. Run the improved (best-practice) system prompt evaluation
    docker run --env-file .env llm-runner python run_improved_prompt.py

    # 3. Run the few-shot evaluation
    docker run --env-file .env llm-runner python run_few_shot_prompt.py

    # 4. Run the automatic prompt optimization
    docker run --env-file .env llm-runner python run_opro_prompt.py
    ```

### Experiments
#### Base Prompt

- **Script**: `run_base_prompt.py`

- **Method**: This script evaluates the model's baseline performance. It uses no system prompt and only appends a simple formatting instruction ("Output final answer after ####. Example: #### ANSWER") to the user's question.
  
- **Output**: `results/eval_base_prompt_{model}.json`

- **Result**: Accuracy = 68.8%
  
#### Improved Prompt

- **Script**: `run_improved_prompt.py`

- **Method**: This experiment uses a detailed, hard-coded system prompt that gives the LLM a specific persona ("expert mathematics tutor") and provides clear, step-by-step instructions for reasoning and formatting the output. This is a "zero-shot" approach, as no examples are provided.

- **Output**: `results/eval_improved_prompt_{model}.json`

- **Result**: Accuracy = 80.4%

#### Few Shot Prompting

- **Script**: `run_few_shot_prompt.py`

- **Method**: This script builds on the "Improved Prompt" by dynamically inserting 2 random examples from the dataset into the prompt for each question being evaluated. It uses the same strong system prompt but adds in-context learning.
  
- **Output**: `results/eval_few_shot_prompt_{model}.json`

- **Result**: Accuracy = 84.4%

#### OPRO-like Automatic Prompt Engineering

- **Script**: `run_opro_prompt.py`

- **Method**: This script implements an iterative optimization loop inspired by OPRO (Optimization by PROmpting).

    1. Begins with the results from the run_base_prompt.py evaluation.

    2. Uses an LLM to generate a set of new, candidate system prompts based on the task examples.

    3. Evaluates each candidate prompt against the entire dataset.

    4. Compares the accuracy of the best new candidate to the current best accuracy. If it's better, it becomes the new "best prompt," and the loop repeats.

    5. The process stops after a fixed number of iterations or if no improvement is found.

- **Output**: `results/eval_opto_summary_{model}.json`

- **Result**: Accuracy 94.0%


### Final Reflection

The results align with my expectations, showing that accuracy improves as prompts are better crafted.

I have observed some outliers in the ground truth data, such as entries listed as “5,000” instead of “5000.” For further optimization, I am considering either correct these inconsistencies in the dataset or use them as examples for in-context learning to achieve higher accuracy.

Automatic prompt engineering produced significantly better results; however, it is extremely costly. Additionally, there is a risk of overfitting to the training set, meaning the improvement might not be substantial compared to cheaper methods.

