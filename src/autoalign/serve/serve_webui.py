# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from autoalign.conversation import Conversation

DEFAULT_CKPT_PATH = "Qwen/Qwen1.5-7B-Chat"


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--template", default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8001, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Max token number of conversation."
    )
    parser.add_argument(
        "--max-new-length", type=int, default=2048, help="Max new token of a response."
    )
    args = parser.parse_args()
    return args


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
    model.generation_config.max_new_tokens = args.max_new_length  # For chat.

    return model, tokenizer


def _chat_stream(model, tokenizer, template, query, history, max_length):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"from": "human", "value": query_h})
        conversation.append({"from": "gpt", "value": response_h})
    conversation.append({"from": "human", "value": query})
    conv = Conversation.from_template(
        template_name=template if template is not None else "chatml"
    )
    conv.fill_in_messages(conv={"conversations": conversation})
    # 适配autoalign
    inputs = conv.get_tokenized_conversation(
        tokenizer=tokenizer, model_max_length=max_length, add_generation_prompt=True
    )
    inputs = torch.tensor(inputs["input_ids"]).unsqueeze(0)
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(
            model,
            tokenizer,
            args.template,
            _query,
            history=_task_history,
            max_length=args.max_length,
        ):
            response += new_text
            _chatbot[-1] = (_query, response)

            yield _chatbot
            full_response = response

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Auto-Alignment Chat Bot: {full_response}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown(
            """\
<p align="center"><img src="https://img1.baidu.com/it/u=1697928215,3074026294&fm=253&fmt=auto&app=120&f=PNG?w=453&h=345" style="height: 100px"/><p>"""
        )
        gr.Markdown("""<center><font size=8>Auto-Alignment Chat Bot Demo</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Auto-Alignment, developed by CIP. \
(本WebUI基于Auto-Alignment打造，实现聊天机器人功能。)</center>"""
        )
        chatbot = gr.Chatbot(
            label="Auto-Alignment Chat Bot", elem_classes="control-height"
        )
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(
            reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )

        gr.Markdown(
            """\
<font size=2>Note: This demo is governed by the original license of Auto-Alignment. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Auto-Alignment的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)"""
        )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()
