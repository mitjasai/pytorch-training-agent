#!/usr/bin/env python3

from dotenv import load_dotenv
import numpy as np
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Load environment variables from `.env`
load_dotenv()

# Set lower and upper bounds for secret number
LOW = 1
HIGH = 128

# Generate secret number
rng = np.random.default_rng()
rints = rng.integers(low=LOW, high=HIGH, size=1)
secret_number = rints[0]


def guess_number(guess: int) -> str:
    """
    Guess the secret number.

    Returns a message indicating whether the guess is correct,
    or if the secret number is lesser or greater than the guess.
    """
    if guess < secret_number:
        return f"The secret number is greater than {guess}."
    elif guess == secret_number:
        return f"Correct, the secret number is {guess}!"
    elif guess > secret_number:
        return f"The secret number is lesser than {guess}."


def main():
    # Print the secret number so we know what it really is
    print(f"The secret number is {secret_number}.\n")

    # Initialize chat model
    chat_model = ChatOpenAI(
        model=os.environ["MODEL"],
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    # Initialize agent
    agent = create_agent(
        model=chat_model,
        tools=[guess_number],
    )

    # Define task prompt for agent
    prompt = (
        "Use the `guess_number` tool to find the secret number "
        f"in the range from {LOW} to {HIGH}."
    )

    # Prompt agent to complete provided task
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="updates",
        version="v2",
    ):
        if chunk["type"] == "updates":
            for step, data in chunk["data"].items():
                print(f"step: {step}")
                print(f"content: {data['messages'][-1].content_blocks}")


if __name__ == "__main__":
    main()
