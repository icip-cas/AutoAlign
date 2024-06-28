"""generate responses"""
from functools import partial

import fire
import torch
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)

from autoalign.conversation import Conversation
from autoalign.utils import read_json, save_json


def get_tokenized_inst(conv, conv_template_name: str, tokenizer: AutoTokenizer):
    """get tokenized inst for response generation"""
    # get conversation template
    conversation = Conversation.from_template(conv_template_name)

    # fill in messages from single conv data point
    conversation.fill_in_messages(conv)

    # get conversation_str and add role starter
    conversation_str = conversation.get_conversation_str()
    inst_str = conversation_str + conversation.role_starts["gpt"]

    # get tokenized inst
    tokenized_inst = tokenizer(inst_str)

    # handle offset for some non-additive tokenizers
    if conversation.offset:
        tokenized_inst.input_ids = tokenized_inst.input_ids[: -conversation.offset]
        tokenized_inst.attention_mask = tokenized_inst.attention_mask[: -conversation.offset]

    return tokenized_inst


@torch.no_grad()
def main(
    data_path: str,
    output_path: str,
    conv_template_name: str,
    model_name_or_path: str,
    # args for inference
    batch_size: int = 16,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    top_p: float = 0.9,
):
    """entrypoint for response generation"""
    # for multi-gpu inference
    accelerator = Accelerator()

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # NB: use eos_token for padding
    tokenizer.pad_token = tokenizer.eos_token
    # set padding_side
    tokenizer.padding_side = "left"

    # get model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map={"": accelerator.process_index},
    )

    # read data
    data = read_json(data_path)

    # for each process
    with accelerator.split_between_processes(data) as data_subset:
        # get dataset
        dataset = Dataset.from_list(data_subset)

        # get tokenized dataset
        dataset = dataset.map(
            partial(
                get_tokenized_inst,
                conv_template_name=conv_template_name,
                tokenizer=tokenizer,
            ),
            remove_columns=list(dataset.features),
            num_proc=8,
        )

        # get data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        # get data_loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )

        # get eos_token_id from conversation template
        conversation_template = Conversation.from_template(conv_template_name)
        gpt_role_end = conversation_template.role_ends["gpt"]
        eos_token_id = tokenizer(gpt_role_end).input_ids[conversation_template.bos_offset]

        # get generation_config
        generation_config = GenerationConfig.from_pretrained(
            model_name_or_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        # generate response
        responses = []
        for batch in tqdm(data_loader):
            batch.to(device)
            model_out = model.generate(
                **batch,
                generation_config=generation_config,
            )
            responses += tokenizer.batch_decode(
                model_out[:, len(batch.input_ids[0]) :],
                skip_special_tokens=True,
            )

    # gather responses
    gathered_responses = gather_object(responses)

    # append responses
    for conv, response in zip(data, gathered_responses):
        conv["conversations"].append(
            {
                "from": "gpt",
                "value": response,
            }
        )

    # save responses
    if accelerator.is_main_process:
        save_json(data, output_path)


if __name__ == "__main__":
    fire.Fire(main)
