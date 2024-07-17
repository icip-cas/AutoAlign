python ./tests/test_conversations.py test_get_tokenized_conversation --template-name "llama-2-chat-keep-system" \
                                --data_path "data/train/pure_data_tag_source_dedup_ins_tag/en/no_deduped_data/inst_ultrachat.json" \
                                --tokenizer-name-or-path "pretrained_models/Llama-2-7b-hf"