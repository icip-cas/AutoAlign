# Evaluation

## Objective evaluation

Objective evaluation involves assessing datasets with standard answers, where processed responses can be directly compared to these standard answers according to established rules and model performances are mesured with quantitative metrics. We utilize the OpenCompass platform to conduct these evaluations.

Usage:
``` bash
autoalign-cli eval --config-path configs/eval.yaml
```
In `eval.yaml`, the `model_path` is the absolute path to the evaluated model or the relative path from the root directory of this repository.

If the `eval_type` is set to be "objective_all" or "objective_core", we use objective evaluation. After running the above command, `autoalign-cli` will call the interface in OpenCompass to conduct an objective dataset evaluation, storing the evaluation results in the `outputs/{model_name}` directory. The raw result will be stored at `outputs/{model_name}/opencompass_log/{opencompass_timestamp}`, in which `{opencompass_timestamp}` is the default name of opencompass output directory of an evaluation. We will summarize and display each evaluation in `outputs/{model_name}/ordered_res.csv`(csv file, easy to process) and `outputs/{model_id}/ordered_res.txt`(formed output, easy to read).

Before starting opencompass, we will check whether new file paths exist, including the config file: `configs/{model_name}.py`, result files: `outputs/{model_name}/ordered_res.csv` and  `outputs/{model_name}/ordered_res.txt`, opencompass logs: `outputs/{model_name}/opencompass_log/`. If one of them exists, you need to choose to continue evaluating or to exit. Continuing may cause overwriting. The overwriting will take the union of the up-to-date results from all datasets.

## Subjective evaluation

Subjective evaluation aims to assess the performance of a model in terms of its ability to conform to human preferences. The gold standard for this kind of evaluation is human preferences, but the cost of annotation is very high.

Usage:
``` bash
autoalign-cli eval --config-path configs/eval.yaml
```
The `eval_type` in config should be set to "subjective" for subjective evalutaion.The mt-bench answers are in `{mt_path}/model_answer/{model_name}.jsonl` and the alpaca-eval answers are in `data/alpaca/{model_name}/{model_name}_outputs.json` according to the default setting. `mt_path` and `model_name` are hyper parameters in `eval.yaml`.


## Safety Evaluation
The safety evaluation assesses several benchmarks including WildGuardTest, HarmBench, ToxiGen, XSTest, and CoCoNot. It aims to evaluate the toxicity of the model's responses and whether it can reasonably refuse.

Usage:
``` bash
autoalign-cli eval --config-path configs/eval_safety.yaml
```
Set `eval_type` to `safety_eval`, which means Safety evaluation.
`per_model_gpu` represents the number of GPUs occupied by a single model worker. Since the data for Safety evaluation is not very large, multiple instances will not be deployed, and only a single instance with multiple GPUs will be used.
The recommended `backend` is `hf`, which is faster.
The path of the model used for evaluation, including `wildguard`, `toxigen roberta`, and `llama guard 3`, can be specified in the `configs/eval_safety.yaml`. If it is not specified, it will be directly loaded from the `hf` cache.  
The results for for one model will be stored in `outputs/{model_name}.safety_eval_metrics.json`, `outputs/{model_name}.safety_eval_all_res.json`. The total results will be stored in `outputs/safety_eval_total_results.tsv`