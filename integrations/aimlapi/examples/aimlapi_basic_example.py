# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Basic text generation example using AIMLAPIChatGenerator."""

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.aimlapi import AIMLAPIChatGenerator


def main() -> None:
    """Generate a response without using any tools."""

    generator = AIMLAPIChatGenerator(
        model="openai/gpt-5-chat-latest",
        generation_kwargs={"models": ["openai/gpt-5-chat-latest", "openai/gpt-4.1"]},
    )

    messages = [
        ChatMessage.from_system("You are a concise assistant."),
        ChatMessage.from_user("Briefly explain what AIMLAPI offers."),
    ]

    reply = generator.run(messages=messages)["replies"][0]

    print(f"assistant response: {reply.text}")  # noqa: T201


if __name__ == "__main__":
    main()
