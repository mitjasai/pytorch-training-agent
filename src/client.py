#!/usr/bin/env python3

import asyncio
from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

# Load environment variables from `.env`
load_dotenv()


async def main():
    # Initialize chat model
    chat_model = ChatOpenAI(
        model=os.environ["MODEL"],
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    # Define task prompt for agent
    prompt = (
        "Generate a secret number using the `generate_number` tool, "
        "then use the `guess_number` tool to find out what it is."
    )

    # Start MCP client for using tools from server
    mcp_client = MultiServerMCPClient(
        {
            "number_guesser": {
                "transport": "stdio",
                "command": "python",
                "args": ["src/server.py"],
            }
        }
    )

    async with mcp_client.session("number_guesser") as session:
        # Load tools from MCP server
        tools = await load_mcp_tools(session)

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
                    print(f"step: {step}")
                    print(f"content: {data['messages'][-1].content_blocks}")


if __name__ == "__main__":
    asyncio.run(main())
