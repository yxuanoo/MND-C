# MND-C

Project Overview
This repository contains all experimental code, datasets, and related configuration files used in [Missing Node Detection Based on Graph Self-supervised Contrastive Learning], aiming to support the reproducibility and transparency of the research.

Content Structure

datasets/: Experimental datasets, including split files for training, validation, and test sets

MND-C/: Core model code folder, including model definitions, training scripts, evaluation scripts, and utility functions
configs/: Experimental parameter configuration files (e.g., learning rate, batch size, etc.)
requirements.txt: Dependency list
Release Plan
The current code and datasets are temporarily stored locally. They are planned to be fully open-sourced on GitHub within 1 week after the paper is officially accepted. This document will be updated to provide the specific access link at that time.

Usage Instructions (After Open-sourcing)
Clone the repository to your local machine
Install dependencies: pip install -r requirements.txt
Run the training script: python MND-C/train.py --config configs/default.yaml
Run the evaluation script: python MND-C/evaluate.py --model_path [model_save_path]
Dataset Description
The datasets used in this study are derived from TUDataset. They were processed through [Select batches of random subgraphs and integrate them separately to form relatively small-scale datasets] before being used in experiments. The open-sourced version has removed all privacy information, retaining only the feature data necessary for the experiments.
