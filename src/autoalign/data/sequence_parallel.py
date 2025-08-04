IGNORE_INDEX = -100

def preprocess_sp_dataset(seq_ids, world_size, sequence_parallel_mode):
    if sequence_parallel_mode == "ulysses":
        step = len(seq_ids) // world_size
        local_values = [seq_ids[s : s + step] for s in range(0, len(seq_ids), step)]
        return local_values
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

def pad_sequence(examples, data_args, tokenizer):
    max_length = data_args.cutoff_len
    input_pad_token_id = tokenizer.pad_token_id
    assert data_args.ignore_pad_token_for_loss
    label_pad_token_id = IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id # 这里填充总是会用IGNORE_INDEX

    for k, v in examples.items():
        if k.endswith("input_ids"):
            pad_token_id = input_pad_token_id
        elif k.endswith("labels"):
            pad_token_id = label_pad_token_id
            # shift labels here
            for i in range(len(v)):
                v[i] = v[i][1:]
        elif k.endswith("attention_mask"):
            pad_token_id = 0
        elif k.endswith("position_ids"):
            pad_token_id = max_length - 1  # pad the max position id
        elif k == "images" or k == "videos":
            pad_token_id = -1
            continue  # TODO: haven't tested multi-modal yet
        else:
            raise NotImplementedError(f"Unexpected dataset key: {k}")
        for i in range(len(v)):
            v[i].extend([pad_token_id] * (max_length - len(v[i]))) # 将每个sequence的token数量补齐到cutoff_len
        examples[k] = v

    return examples

# sp for Sequence Parallel
def sp_split(examples, model_args):
    for k, v in examples.items():
        chunks = list()
        for row in v:
            # if k.endswith("attention_mask"):
            #     chunks.extend([row] * model_args.sequence_parallel_size)
            # elif row is None:
            #     chunks.extend([None] * model_args.sequence_parallel_size)
            # else:
            chunks.extend(
                preprocess_sp_dataset(row, model_args.sequence_parallel_size, model_args.sequence_parallel_mode)
            )
        examples[k] = chunks # 将一个sequence[] 拆分成了model_args.sequence_parallel_size个小sequence[[],[],...]
    return examples
