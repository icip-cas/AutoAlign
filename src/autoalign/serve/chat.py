import time
import torch
from threading import Thread
from autoalign.conversation import Role, Conversation
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def _chat_stream(model, tokenizer, conversation, max_length):
    # 适配autoalign
    inputs = conversation.get_tokenized_conversation(tokenizer=tokenizer, model_max_length=max_length, add_generation_prompt=True)
    inputs.data = {k: torch.tensor(v).unsqueeze(0) for k, v in inputs.items()}
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def chat_loop(
    args,
    chatio,
    debug: bool = True,
    history: bool = True,
):
    # Model
    def _load_model_tokenizer(args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path,
            resume_download=True,
        )

        if args.cpu_only:
            device_map = "cpu"
        else:
            device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map=device_map,
            resume_download=True,
        ).eval()

        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id
        # TODO: refine the style
        model.generation_config.max_new_tokens = args.max_new_tokens if args.max_new_tokens else model.generation_config.max_new_tokens
        model.generation_config.temperature = args.temperature if args.temperature else model.generation_config.temperature
        model.generation_config.do_sample = args.do_sample if args.do_sample else model.generation_config.do_sample
        model.generation_config.top_p = args.top_p if args.top_p else model.generation_config.top_p
        model.generation_config.repetition_penalty = (
            args.repetition_penalty if args.repetition_penalty else model.generation_config.repetition_penalty
        )
        return model, tokenizer

    model, tokenizer = _load_model_tokenizer(args)

    # Chat
    def new_chat(args) -> Conversation:
        conv = Conversation.from_template(template_name=args.template if args.template is not None else "chatml")
        return conv

    def reload_conv(conv, chatio):
        """
        Reprints the conversation from the start.
        """
        for message in conv.messages[conv.template.offset :]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    conv = None

    while True:
        if not history or not conv:
            conv = new_chat(args)

        try:
            inp = chatio.prompt_for_input(Role.HUMAN.value)
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break
        elif inp == "!!reset":
            print("resetting...")
            conv = new_chat(args)
            continue
        elif inp == "!!remove":
            print("removing last message...")
            if len(conv.messages) > conv.template.offset:
                if conv.messages[-1][0] == Role.HUMAN or Role.ASSISTANT:
                    conv.pop_message(-1)
                else:
                    # Shouldn't happen in normal circumstances
                    print("System message cannot be removed.")
                reload_conv(conv, chatio)
            else:
                print("No messages to remove.")
            continue
        elif inp == "!!regen":
            print("regenerating last message...")
            if len(conv.messages) > conv.template.offset:
                # Assistant
                if conv.messages[-1][0] == Role.ASSISTANT:
                    conv.pop_message(-1)
                # User
                if conv.messages[-1][0] == Role.HUMAN and len(conv.messages) > conv.template.offset:
                    reload_conv(conv, chatio)
                    # Set inp to previous message
                    inp = conv.pop_message(-1)[1]
                else:
                    # Shouldn't happen in normal circumstances
                    print("No user message to regenerate from.")
                    continue
            else:
                print("No messages to regenerate.")
                continue
        # TODO: save and load. The conversation is not supported to be serialized!
        # elif inp.startswith("!!save"):
        #     args = inp.split(" ", 1)

        #     if len(args) != 2:
        #         print("usage: !!save <filename>")
        #         continue
        #     else:
        #         filename = args[1]

        #     # Add .json if extension not present
        #     if not "." in filename:
        #         filename += ".json"

        #     print("saving...", filename)
        #     with open(filename, "w") as outfile:
        #         json.dump(conv.get_attributes(), outfile)
        #     continue
        # elif inp.startswith("!!load"):
        #     args = inp.split(" ", 1)

        #     if len(args) != 2:
        #         print("usage: !!load <filename>")
        #         continue
        #     else:
        #         filename = args[1]

        #     # Check if file exists and add .json if needed
        #     if not os.path.exists(filename):
        #         if (not filename.endswith(".json")) and os.path.exists(
        #             filename + ".json"
        #         ):
        #             filename += ".json"
        #         else:
        #             print("file not found:", filename)
        #             continue

        #     print("loading...", filename)
        #     with open(filename, "r") as infile:
        #         new_conv = json.load(infile)

        #     conv = Conversation.from_template(new_conv["template"]["name"])
        #     conv._system_message = new_conv["system_message"]
        #     # conv.set_system_message(new_conv["system_message"])
        #     conv.messages = new_conv["messages"]
        #     reload_conv(conv)
        #     continue

        conv.append_message(Role.HUMAN, inp)
        try:
            chatio.prompt_for_output(Role.ASSISTANT.value)
            output_stream = _chat_stream(model, tokenizer, conv, args.max_new_tokens)
            t = time.time()
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            conv.append_message(Role.ASSISTANT, outputs.strip())

            if debug:
                num_tokens = len(tokenizer.encode(outputs))
                msg = {
                    "conv_template": conv.name,
                    "prompt": conv.get_conversation_str(add_generation_prompt=True),
                    "outputs": outputs,
                    "speed (token/s)": round(num_tokens / duration, 2),
                }
                print(f"\n{msg}\n")

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)
