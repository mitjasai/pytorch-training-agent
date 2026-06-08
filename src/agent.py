#!/usr/bin/env python3

import argparse
import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# Load environment variables from `.env`
load_dotenv()


async def main(prompt: str):
    # Initialize chat model
    chat_model = ChatOpenAI(
        model=os.environ["MODEL"],
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    # Load tools from MCP server
    mcp_client = MultiServerMCPClient(
        {
            "pytorch_trainer": {
                "transport": "stdio",
                "command": "python3",
                "args": ["src/mcp_server.py"],
            }
        }
    )
    tools = await mcp_client.get_tools()

    # Initialize agent
    agent = create_agent(model=chat_model, tools=tools)

    # Prompt agent to complete provided task
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="updates",
        version="v2",
    ):
        if chunk["type"] == "updates":
            for step, data in chunk["data"].items():
                for block in data["messages"][-1].content_blocks:
                    if step == "model":
                        if block["type"] == "tool_call":
                            print(
                                f"Tool call: {{'name': {block['name']}, "
                                f"'args': {block['args']}}}"
                            )
                        if block["type"] == "text":
                            print(block["text"])

                    elif step == "tools":
                        if block["type"] == "text":
                            print(f"Tool response: {block['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    args = parser.parse_args()

    asyncio.run(main(**vars(args)))
