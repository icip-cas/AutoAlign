from einops import rearrange
import torch


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def first(arr):
    return arr[0]


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def identity(t, *args, **kwargs):
    return t


def prompt_mask_from_len(length, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device=device) < rearrange(length, "... -> ... 1")


def cast_tuple(t, length=1, validate=False):
    out = t if isinstance(t, tuple) else ((t,) * length)
    assert not validate or len(out) == length
    return out


def extract_prompts(answer):
    # find all the prompts between <task> </task> brackets
    print("=" * 80)
    print("Extracting prompts...")
    print(answer)
    print("=" * 80)

    prompts = []
    while True:
        pattern = "<task>"
        start = answer.find(pattern)
        if start == -1:
            break
        end = answer.find("</task>")
        if end == -1:
            break
        prompts.append(answer[start + len(pattern) : end])
        answer = answer[end + len("</task>") :]

    print("Prompts extracted:")
    print(prompts)
    return prompts
