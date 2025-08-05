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
The results for one model will be stored in `outputs/{model_name}/safety_eval_metrics.json`, `outputs/{model_name}/safety_eval_all_res.json`. The total results will be stored in `outputs/safety_eval_total_results.tsv`

## Reasoning model Evaluation
Reasoning model evaluation focuses on assessing a model's capability to perform complex reasoning tasks, including AIME24, AIME25, LiveCodeBench, IFEval, MBPP+, and GPQA, aimed at evaluating the reasoning model's abilities in mathematics, coding, knowledge-based question answering, and instruction following.

Usage:
``` bash
autoalign-cli eval --config-path configs/eval_cot.yaml
```

In `eval_cot.yaml`, set `eval_type` to `cot_eval` for Reasoning model evaluation, `per_model_gpu` to define the number of GPUs per model worker, `dp_gpu` to configure data parallelism, `n_samples` to specify the number of samples in evaluation, and `max_length` to set the maximum response length.

The results for AIME24 and AIME25 will be stored in `outputs/{data_name}/{model_name}_{split}_{prompt_type}_{num_test_sample}_seed{seed}_t{temperature}_s{start}_e{end}_{prompt_type}_metrics.json`.

The results for LiveCodeBench will be stored in `outputs/{model_name}/Scenario.codegeneration_{num_test_sample}_{args.temperature}_eval.json`.

The results for IFEval, MBPP+, and GPQA will be stored in `outputs/{model_name}`.

It must be noted that:
1. To modify the `path` of `mbpp_plus_datasets` in `opencompass/configs/datasets/mbpp_plus/mbpp_plus_gen_0b836a.py`, change it to `'opencompass/mbpp_plus'`. Additionally, add the following to `DATASETS_MAPPING` in `opencompass/opencompass/utils/datasets_info.py`:
```python
'opencompass/mbpp_plus': {
    'ms_id': '',
    'hf_id': '',
    'local': './data/mbpp_plus/mbpp_plus.jsonl',
}
```
2. The `score` method in the MBPPEvaluator class for MBPP evaluation is incompatible with the `evalplus` library. The input to the `evaluate` function in the `evalplus` library needs to be modified from `self.eval(flags)` to `self.eval(**flags)`. Additionally, the parameters `base_only`, `i_just_wanna_run`, and `mini` in `flags` should be set to `False`. The output of the `evaluate` function also needs modification. Originally, it had no output; now, it should return the `pass_at_k` results.
3. If you need to evaluate a new model in LiveCodeBench, you should modify `AutoAlign/src/autoalign/eval/livecodebench/lcb_runner/lm_styles.py`:
- To add a different LMStyle, first add the model type in the LMStyle class, then add the model's LanguageModel class in LanguageModelList.
- To add a different model under the same LMStyle, directly add the model's LanguageModel class in LanguageModelList.
4. Due to the potentially varying performance of the reasoning model under few-shot and zero-shot settings, to modify the evaluation templates, you can follow these steps:
For AIME24 and AIME25 evaluation templates, modify `PROMPT_TEMPLATES` in `AutoAlign/src/autoalign/eval/math_eval/utils.py`.  
For LiveCodeBench evaluation templates, modify the `format_prompt_generation` function in `AutoAlign/src/autoalign/eval/livecodebench/lcb_runner/prompts/code_generation.py`.  
For templates evaluated in OpenCompass, locate the respective Python file for each dataset in `AutoAlign/opencompass/opencompass/configs/datasets` and modify the `template` in `{data_name}_infer_cfg`.

Evaluation Reference Results:
| Model       | AIME24 | AIME25 | LCB  | MBPP+ | GPQA  | IFEval Prompt-level-strict-accuracy | IFEval Inst-level-strict-accuracy | IFEval Prompt-level-loose-accuracy | IFEval Inst-level-loose-accuracy |
|-------------|--------|--------|------|-------|-------|-------------------------------------|------------------------------------|-------------------------------------|----------------------------------|
| distill-1.5b | 27.3   | 21.6   | 17.2 | 1.06  | 27.27 | 26.99                               | 41.13                              | 28.10                               | 42.81                            |

