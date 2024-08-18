from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from autoalign.model_merging.averaging import average_merging


def parse_args(args: list[str]):

    parser = ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True)
    parser.add_argument("--merged_model_path", type=str, required=True)
    parser.add_argument("--merged_config_path", type=str)
    parser.add_argument("--merging_method", type=str, required=True)
    parser.add_argument("--average_weight", nargs="+", type=float)
    parser.add_argument("--exclude_param_names_regex", nargs="+", type=str)

    args = parser.parse_args(args)
    return args


def run_merge(args):
    args = parse_args(args=args)
    assert (
        args.model_paths and len(args.model_paths) > 0
    ), '"Paths" should be a list containing valid hf model paths.'
    if not args.merged_config_path:
        merged_config_path = args.model_paths[0]
    else:
        merged_config_path = args.merged_config_path
    models = []
    for path in args.model_paths:
        models.append(AutoModelForCausalLM.from_pretrained(path))

    if args.merging_method == "average":
        merged_weight = average_merging(
            models, args.average_weight, args.exclude_param_names_regex
        )
    else:
        raise NotImplementedError(
            "Merging method is not implemented. Please contact developer."
        )

    merged_tokenizer = AutoTokenizer.from_pretrained(merged_config_path)
    merged_model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(merged_config_path)
    )
    merged_model.load_state_dict(merged_weight)
    merged_model.save_pretrained(args.merged_model_path)
    merged_tokenizer.save_pretrained(args.merged_model_path)
