# Capstone: A Comparative Study of Monosemanticity Across Language Models

This project investigates how innate monosemantic features are distributed across different language model architectures, specifically comparing Google's Gemma, Meta's LLaMA, and OpenAI's GPT-2 models. The research builds upon recent work in neural network interpretability and feature extraction.

## Project Overview

We aim to understand how different model architectures and sizes influence the development of monosemantic features - neural network components that consistently track specific, interpretable concepts.

## Installation & Setup

```bash
pip install -r requirements.txt
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
```
Note: 
- You can create a Hugging Face token at [link](https://huggingface.co/docs/hub/security-tokens).
- This requires access to a GPU with CUDA installed.

## Usage

```
usage: main.py [-h] [-em EXECUTION_MODE] [-rn RUN_NAME] [-lm LLM_MODEL] [-dm DATA_MODEL] [-bs BATCH_SIZE] [-lr LEARNING_RATE] [-me MAXIMUM_EPOCHS] [-op OUTPUT_PATH] [-rs RANDOM_SEED] [-ocs OVER_COMPLETE_SIZE]

options:
  -h, --help            show this help message and exit
  -em EXECUTION_MODE, --execution-mode EXECUTION_MODE
  -rn RUN_NAME, --run-name RUN_NAME
  -lm LLM_MODEL, --llm-model LLM_MODEL
  -dm DATA_MODEL, --data-model DATA_MODEL
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
  -me MAXIMUM_EPOCHS, --maximum-epochs MAXIMUM_EPOCHS
  -op OUTPUT_PATH, --output-path OUTPUT_PATH
  -rs RANDOM_SEED, --random-seed RANDOM_SEED
  -ocs OVER_COMPLETE_SIZE, --over-complete-size OVER_COMPLETE_SIZE

```

Example:
```bash
python main.py --execution-mode train --run-name run1 -lm gpt -dm wikitext --maximum-epoch 100 -ocs 5
```

## Contributors

- Einstein Oyewole (eo2233@nyu.edu)
- Alon Florentin (abf386@nyu.edu)

## References

1. Henighan et al. (2023). "Superposition, memorization, and double descent"
2. Templeton et al. (2024). "Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet"
3. Bricken et al. (2023). "Towards monosemanticity: Decomposing language models with dictionary learning"