# SSO

## Overview

SSO (Self-Supervised Optimization) is a training algorithm that combines generator-based data augmentation with Direct Preference Optimization (DPO). The algorithm works in five main phases:

1. **Build Principle**: We guide the model to generate a pair of opposing principles for a given instruction through a carefully designed principle_prompt. For generating principles, we can optionally use a very powerful model - the original SSO paper used qwen2.5-72b-instruct;
2. **Build SSO Training Data**: The generated pair of opposing principles are respectively used as system prompts, combined with the instruction and the original instruction, all input to the policy model to sample three responses;
3. **SSO Training**: We use the SSO algorithm to train the policy model, obtaining the generator model;
4. **Build DPO Training Data**: The pair of opposing principles are respectively used as system prompts combined with instructions and input to the generator model to sample two responses;
5. **DPO Training**：We use the generated preference data and DPO algorithm to train the policy model, obtaining the final model.

## Environment settings
transformers==4.48.3

## Workflow

### 1. Build Principles Data
```bash
bash build_principles.bash
```

*Input Example:*
```
[
  {
    "prompt": "how can i develop a habit of drawing daily",
    "uuid": "cb71928c-1b78-4ab5-9db9-f6bdd128d942"
  }
]
```

*Output Example:*
```
[
  {
    "prompt": "how can i develop a habit of drawing daily",
    "uuid": "cb71928c-1b78-4ab5-9db9-f6bdd128d942",
    "principles": [
      {
        "good_principle": "",
        "bad_principle": ""
      }
    ]
  }
]
```

### 2. Build SSO Training Data
```bash
bash build_sso_data.bash
```

*Input Example:*
```
[
  {
    "prompt": "how can i develop a habit of drawing daily",
    "uuid": "cb71928c-1b78-4ab5-9db9-f6bdd128d942",
    "principles": [
      {
        "good_principle": "",
        "bad_principle": ""
      }
    ]
  }
]
```

*Output Example:*
```
[
  {
    "prompt": "",
    "uuid": "de3efad1-9b8a-4ea7-9d54-de406674b668",
    "p_system": "",
    "n_system": "",
    "system": "",
    "chosen": [
      {
        "value": "",
        "from": "human"
      },
      {
        "value": "",
        "from": "gpt"
      }
    ],
    "rejected": [
      {
        "value": "",
        "from": "human"
      },
      {
        "value": "",
        "from": "gpt"
      }
    ],
    "raw": [
      {
        "value": "",
        "from": "human"
      },
      {
        "value": "",
        "from": "gpt"
      }
    ],
    "history": []
  }
]
```

### 3. Execute SSO Training
```bash
bash sso.bash
```

### 4. Build DPO Data
```bash
bash build_dpo_data.bash
```

*Input Example:*
```
[
  {
    "prompt": "how can i develop a habit of drawing daily",
    "uuid": "cb71928c-1b78-4ab5-9db9-f6bdd128d942",
    "principles": [
      {
        "good_principle": "",
        "bad_principle": ""
      }
    ]
  }
]
```

*Output Example:*
```
[
  {
    "prompt": "how can i develop a habit of drawing daily",
    "chosen": [
      {
        "value": "how can i develop a habit of drawing daily",
        "from": "human"
      },
      {
        "value": "",
        "from": "gpt"
      }
    ],
    "rejected": [
      {
        "value": "how can i develop a habit of drawing daily",
        "from": "human"
      },
      {
        "value": "",
        "from": "gpt"
      }
    ],
    "conversations": [
      {
        "value": "how can i develop a habit of drawing daily",
        "from": "human"
      },
      {
        "value": "",
        "from": "gpt"
      }
    ],
    "prompt_id": "cb71928c-1b78-4ab5-9db9-f6bdd128d942"
  }
]
```

### 5. Execute DPO Training
```bash
bash dpo.bash
```

### Quick Start
```bash
bash run_sso.bash
```

## Implementation Results
| Model | 1c win rate (gpt4o) | 1c win rate (gpt4) |
|-------|---------------------|---------------------|
| Using qwen2_7b_ins to generate principle |  |  |
| qwen2_7b_ins | 15.07 | 27.65 |
| qwen2_7b_ins_rlcd_sys | 22.42 | 32.26 |
| qwen2_7b_ins_sso | 25.27 | 34.87 |
| Using qwen2.5_72b_ins to generate principle |  |  |
| llama3_8b_sft | 17.11 | 25.26 |
| llama3_8b_sft_rlcd_sys | 29.58 | 37.57 |
| llama3_8b_sft_sso | 33.31 | 42.34 |

The experimental results use the train_prefs-00000-of-00001.parquet data from HuggingFaceH4/ultrafeedback_binarized. Before using this data, simple preprocessing is required. You can run the following script for processing.
```bash
bash convert_parquet_to_json.bash
```