# Code and Dataset Documentation

## Project Overview
This repository contains all experimental code, datasets, and related configuration files used in [***Missing Node Detection** **Based on Graph Self-supervised Contrastive Learning***], aiming to support the reproducibility and transparency of the research.

## Content Structure
- `datasets/`: Experimental datasets, including split files for training, validation, and test sets
- `MND-C/`: Core model code folder, including model definitions, training scripts, evaluation scripts, and utility functions
- `configs/`: Experimental parameter configuration files (e.g., learning rate, batch size, etc.)
- `requirements.txt`: Dependency list

## Usage Instructions (After Open-sourcing)
1. Clone the repository to your local machine
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training script: `python MND-C/main.py --config configs/default.yaml`
4. Run the evaluation script: `python MND-C/test.py --model_path [model_save_path]`

## Dataset Description
The datasets used in this study are derived from [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/?utm_source=chatgpt.com). They were processed through [Select batches of random subgraphs and integrate them separately to form relatively small-scale datasets] before being used in experiments. The open-sourced version has removed all privacy information, retaining only the feature data necessary for the experiments.



