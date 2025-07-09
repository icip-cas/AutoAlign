---
license: cc
tags:
- code
- code generation
pretty_name: LiveCodeBench
size_categories:
- n<1K
---
## LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code

<p align="center">
    <a href="https://livecodebench.github.io/">🏠 Home Page</a> •
    <a href="https://github.com/LiveCodeBench/LiveCodeBench">💻 GitHub Repository </a> •
    <a href="https://livecodebench.github.io/leaderboard.html">🏆 Leaderboard</a> •
    <a href="https://arxiv.org/abs/2403.07974">📄 Paper </a>
</p>

![LiveCodeBench](images/lcb.png)

## Change Log
Since LiveCodeBench is a continuously updated benchmark, we provide different versions of the dataset. Particularly, we provide the following versions of the dataset:
- `release_v1`: The initial release of the dataset with problems released between May 2023 and Mar 2024 containing 400 problems.
- `release_v2`: The updated release of the dataset with problems released between May 2023 and May 2024 containing 511 problems.
- `release_v3`: The updated release of the dataset with problems released between May 2023 and Jul 2024 containing 612 problems.
- `release_v4`: The updated release of the dataset with problems released between May 2023 and Sep 2024 containing 713 problems.
- `release_v5`: The updated release of the dataset with problems released between May 2023 and Jan 2025 containing 880 problems.

You can use the `version_tag` argument to load the desired version of the dataset. Additionally, you can use version tags like `v1`, `v2`, `v1_v3`, `v4_v5` to get the problems released in a specific version.

## Dataset Description

LiveCodeBench is a "live" updating benchmark for holistically evaluating code related capabilities of LLMs. 
Particularly, it evaluates LLMs across a range of capabilties including code generation, self-repair, test output prediction, and code execution. 
This is the code generation scenario of LiveCodeBench. It is also used for evaluating self-repair using test case feedback.

LiveCodeBench problems are collected from competition programming websites with particular focus on maintaining problem quality, test case quality, and problem difficulty diversity. 
This scenario currently hosts over 500 problems from LeetCode, AtCoder, and Codeforces.
Each problem instance consists of a problem description, input/output examples, and hidden test cases. 
Additionally, every problem is tagged with its difficulty level and release date, which allows measuring model performance across different time windows. 
The goal is to generate a correct and efficient solution for each problem instance.

The initial code_generation dataset included a larger number of test cases which leads to a substantially large dataset size. This (lite) version has pruned and sampled tests while trying to ensure similar performances with the original dataset. Going forward, livecodebench will be using this lite version for code generation evaluations.

## Usage
You can use the dataset by loading it from the Hugging Face datasets library. Additionally, the version tag "release_v1" is used to specify the (temporal) version of the dataset. "v1" corresponds to the initial release of the dataset and "release_v2" is the second version.

```python
from datasets import load_dataset
lcb_codegen = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2")
```