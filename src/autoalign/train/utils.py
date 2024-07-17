from autoalign.conversation import Conversation

def configure_model(conv_template_name, tokenizer, model):
    """ specify eos token and bos token for model and tokenizer based on conversation template """
    conversation = Conversation.from_template(conv_template_name)
    eos_token = conversation.role_ends["gpt"].strip()
    eos_token_id = tokenizer(eos_token).input_ids[-1]
    # print(f"{tokenizer(eos_token)=} {eos_token=} {eos_token_id=} {tokenizer.decode([eos_token_id])=}")

    assert eos_token == tokenizer.decode([eos_token_id]), "eos token is not a valid token"
    tokenizer.eos_token_id = eos_token_id
    tokenizer.eos_token = eos_token
    
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.config.use_cache = False # disable cache for training
    
    return None