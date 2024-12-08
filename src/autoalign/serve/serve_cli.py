import argparse

from autoalign.serve.utils import chat_loop


class SimpleChatIO:
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        output_text = []
        for outputs in output_stream:
            output_text.append(outputs)
        print("".join(output_text[pre:]), flush=True)
        return "".join(output_text)

    def print_output(self, text: str):
        print(text)


def main(args):
    chatio = SimpleChatIO(args.multiline)
    try:
        chat_loop(
            args,
            chatio,
            debug=args.debug,
            history=not args.no_history,
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add_model_args(parser)
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint name or path",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--template", type=str, required=True, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Max token number of conversation."
    )
    args = parser.parse_args()
    main(args)
