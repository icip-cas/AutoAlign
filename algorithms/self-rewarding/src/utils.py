import subprocess
import re
from tqdm import tqdm
import string
import json
from pathlib import Path
import transformers


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def post_process_instructs(responses):
    if responses is None:
        return []
    instructions = []
    for response in tqdm(responses):
        raw_instructions = re.split(r"\n\d+\s?\.\s?", response)
        for inst in raw_instructions:
            inst = re.sub(r"\s+", " ", inst).strip()
            inst = inst.strip().capitalize()
            if inst == "":
                continue
            # filter out too short or too long instructions
            if len(inst.split()) <= 3 or len(inst.split()) > 150:
                continue
            # filter based on keywords that are not suitable for language models.
            if any(
                find_word_in_string(word, inst)
                for word in [
                    "image",
                    "images",
                    "graph",
                    "graphs",
                    "picture",
                    "pictures",
                    "file",
                    "files",
                    "map",
                    "maps",
                    "draw",
                    "plot",
                    "go to",
                ]
            ):
                continue
            # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
            # And it's a bit comfusing whether the model need to write a program or directly output the result.
            # Here we filter them out.
            # Note this is not a comprehensive filtering for all programming instructions.
            if inst.startswith("Write a program"):
                continue
            # filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # filter those starting with non-english character
            if not inst[0].isascii():
                continue
            instructions.append(inst)
    return instructions


def identity(t, *args, **kwargs):
    return t


def get_available_gpu_list():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
        )
        output = result.decode("utf-8")
        gpu_list = output.strip().split("\n")
        return gpu_list
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while getting GPU information: {e}")
        return 0


def dump_jsonlines(obj, filepath, **kwargs):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wt", encoding="utf-8") as fout:
        for d in obj:
            line_d = json.dumps(d, ensure_ascii=False, **kwargs)
            fout.write("{}\n".format(line_d))


def load_jsonlines(filepath, **kwargs):
    data = list()
    with open(filepath, "rt", encoding="utf-8") as fin:
        for line in fin:
            line_data = json.loads(line.strip())
            data.append(line_data)
    return data


def load_json(filepath, **kwargs):
    with open(filepath, "rt", encoding="utf-8") as fin:
        data = json.load(fin, **kwargs)
    return data


def dump_json(obj, filepath, **kwargs):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wt", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, **kwargs)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
