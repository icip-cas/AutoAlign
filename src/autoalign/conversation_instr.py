from .conversation import Conversation, Role


class InstructionGenerateConversation(Conversation):
    def swap_eos_token(self):
        temp = self.template.role_ends[Role.HUMAN]
        self.template.role_ends[Role.HUMAN] = self.template.role_ends[Role.ASSISTANT]
        self.template.role_ends[Role.ASSISTANT] = temp

    def get_conversation_str(self, add_generation_prompt: str = "assistant") -> str:
        """Get full conversation str"""
        if self.template.strategy:
            return self.template.strategy.get_conversation_str(
                self.messages, self.template.get_attributes(), add_generation_prompt
            )

        ret = ""
        for role, message in self.messages:
            ret += (
                self.template.role_starts[role]
                + message
                + self.template.role_ends[role]
            )
        if add_generation_prompt:
            if add_generation_prompt == "assistant":
                ret += self.template.role_starts[Role.ASSISTANT]
            elif add_generation_prompt == "human":
                ret += self.template.role_starts[Role.HUMAN]
        return ret
