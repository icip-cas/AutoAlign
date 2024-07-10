import argparse

def generate_model_config(model_name, model_path, batch_size, num_gpus):
    return f"""models = [
    dict(type=VLLM,
    path="{model_path}",
    max_seq_len=2048,
    model_kwargs=dict(tensor_parallel_size={num_gpus},enforce_eager=True),
    generation_kwargs=dict(temperature=0),
    abbr="{model_name}",
    max_out_len=256,
    batch_size={batch_size},
    run_cfg=dict(num_gpus={num_gpus}, num_procs=1))
]"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--eval_type', type=str, required=True)
    args = parser.parse_args()
    model_name = args.model_name
    model_path = args.model_path
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    eval_type = args.eval_type
    if eval_type not in ["core", "full"]:
        print("Invalid eval type")
        raise ValueError
    elif eval_type == "core":
        with open("../opencompass/core_config.py") as f:
            config = f.read()
    else:
        with open("../opencompass/full_config.py") as f:
            config = f.read()
    config = config.replace("### MODEL", generate_model_config(model_name, model_path, batch_size, num_gpus))
    with open(f"../opencompass/tmp_config/{model_name}_{eval_type}.py", "w") as f:
        f.write(config)