import os
from openai import OpenAI
import tiktoken

class ContextManager:
    """
    Production-grade conversation context manager
    with automatic summarization compression.
    """

    def __init__(
        self,
        model="nvidia/nemotron-3-super-120b-a12b:free",
        context_limit=128000,
        reserved_output_tokens=2000,
        keep_last_n=6
    ):

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
        self.model = model

        self.context_limit = context_limit
        self.reserved_output_tokens = reserved_output_tokens
        self.max_input_tokens = context_limit - reserved_output_tokens

        self.keep_last_n = keep_last_n

        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # self.encoder = tiktoken.encoding_for_model(model)
        self.encoder = tiktoken.get_encoding("cl100k_base")

    # -------------------------------------------------
    # Token counting
    # -------------------------------------------------

    def count_tokens(self, messages=None):
        if messages is None:
            messages = self.messages

        tokens = 0
        for m in messages:
            tokens += len(self.encoder.encode(m["content"]))
        return tokens

    # -------------------------------------------------
    # Add message
    # -------------------------------------------------

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    # -------------------------------------------------
    # Context compression
    # -------------------------------------------------

    def compress_history(self):

        print("⚠ Context too large. Compressing history...")

        # Split messages
        system_msg = self.messages[0]
        recent_msgs = self.messages[-self.keep_last_n:]
        old_msgs = self.messages[1:-self.keep_last_n]

        if not old_msgs:
            return

        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation briefly but keep all "
                    "important facts, decisions, and context."
                )
            },
            {
                "role": "user",
                "content": str(old_msgs)
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=summary_prompt,
            max_tokens=500
        )

        summary = response.choices[0].message.content

        # Rebuild message stack
        self.messages = [
            system_msg,
            {
                "role": "system",
                "content": "Conversation summary so far:\n" + summary
            }
        ] + recent_msgs

    # -------------------------------------------------
    # Ensure context limit
    # -------------------------------------------------

    def ensure_limit(self):

        while self.count_tokens() > self.max_input_tokens:
            self.compress_history()

    # -------------------------------------------------
    # Call LLM
    # -------------------------------------------------

    def chat(self, user_input):

        self.add_message("user", user_input)

        self.ensure_limit()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.reserved_output_tokens
        )

        answer = response.choices[0].message.content

        self.add_message("assistant", answer)

        usage = response.usage

        print(
            f"Prompt Tokens: {usage.prompt_tokens}, "
            f"Completion Tokens: {usage.completion_tokens}, "
            f"Total Tokens: {usage.total_tokens}"
        )

        return answer
