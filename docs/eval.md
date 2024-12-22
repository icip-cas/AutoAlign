## Objective evaluation

Objective evaluation involves assessing datasets with standard answers, where processed responses can be directly compared to these standard answers according to established rules and model performances are mesured with quantitative metrics. We utilize the OpenCompass platform to conduct these evaluations.

Usage:
``` bash
autoalign-cli eval --config-path configs/eval.yaml
```
In `eval.yaml`, the `model_path` is the absolute path to the evaluated model or the relative path from the root directory of this repository.

If the `eval_type` After running the above command, `autoalign-cli` will call the interface in OpenCompass to conduct an objective dataset evaluation, storing the evaluation results in the `outputs/{model_name}` directory. The raw result will be stored at `outputs/{model_name}/opencompass_log/{opencompass_timestamp}`, in which `{opencompass_timestamp}` is the default name of opencompass output directory of an evaluation. We will summarize and display each evaluation in `outputs/{model_name}/ordered_res.csv`(csv file, easy to process) and `outputs/{model_id}/ordered_res.txt`(formed output, easy to read).

Before starting opencompass, we will check whether new file paths exist, including the config file: `configs/{model_name}.py`, result files: `outputs/{model_name}/ordered_res.csv` and  `outputs/{model_name}/ordered_res.txt`, opencompass logs: `outputs/{model_name}/opencompass_log/`. If one of them exists, you need to choose to continue evaluating or to exit. Continuing may cause overwriting. The overwriting will take the union of the up-to-date results from all datasets.
