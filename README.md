<h1 align="center">
  ResearchQA
</h1>
<p align="center">
  🌐 <a href="https://cylumn.com/ResearchQA">Website</a> | 
  📄 <a href="">Paper</a> | 
  🗂️ <a href="https://huggingface.co/datasets/realliyifei/ResearchQA">Dataset</a>
</p>

ResearchQA is designed to evaluate scholarly question answering across 75 fields, using questions and rubrics mined from survey papers. The dataset consists of 3,750 questions in the test set, 703 in the validation set, and  a total of 21,414 questions. Both the questions and rubrics have been validated by 31 Ph.D. level annotators across 8 fields.

## Introduction

This repository contains the necessary scripts to download the ResearchQA dataset and compute rubric coverage scores.

*   `download_researchqa.py`: Downloads the ResearchQA dataset from the Hugging Face Hub. You can specify different data splits.
*   `compute_coverage.py`: Computes rubric coverage scores for a given set of responses to the ResearchQA questions. This script uses gpt-4.1-mini to evaluate how well a response covers the rubric items associated with a question.

## Installation

```bash
# Clone the repository
git clone https://github.com/realliyifei/ResearchQA.git
cd ResearchQA
# Create a new conda environment and install dependencies
conda create -n researchqa python=3.10
conda activate researchqa
pip install -r requirements.txt
```

## Configuration

```bash
# OpenAI API Key
export OPENAI_API_KEY="your-openai-api-key"
# Hugging Face Hub Token (if download automatically)
export HUGGING_FACE_HUB_TOKEN="your-hugging-face-token"
```

## Usage

### Download the Dataset

You can download different splits of the [ResearchQA dataset](https://huggingface.co/datasets/realliyifei/ResearchQA).

```bash
python download_researchqa.py --split test.json
```

The available splits are `test.json`, `valid.json`, and `full.json`. 

### Compute Rubric Coverage

To compute the coverage score, you need the dataset file (e.g., `test.json`) and a JSON file containing the model's responses.

The response file should be a map where keys are item IDs and values are objects containing an "answer" field.
Example `responses.json`:
```json
{
  "id_1": {
    "answer": "This is the model's answer to question 1."
  },
  "id_2": {
    "answer": "This is the model's answer to question 2."
  }
}
```

## Citation

If you find our work helpful, please cite us with
```bibtex
@article{yifei2025researchqa,
    title={Research{QA}: Evaluating Scholarly Question Answering Across 75 Fields with Survey-Mined Questions and Rubrics},
    author={Yifei, Li S. and Chang, Allen and Malaviya, Chaitanya and Yatskar, Mark},
    journal={arXiv preprint},
    year={2025}
}
```