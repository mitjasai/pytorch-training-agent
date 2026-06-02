# LLM agent with LangChain and MCP

This repository contains a simple example of implementing an LLM agent using LangChain and MCP. The
agent calls the `generate_number` tool to generate a random number, and then the `guess_number`
tool until it guesses the correct number.

While the agent may not be especially useful in itself, it can be easily adapted for other, more
meaningful tasks.

## Usage

Copy the included `.env.example` to `.env` and replace the dummy values in the latter file with
real ones.

```bash
cp .env.example .env
```

Create a Python virtual environment and install dependencies.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the client script.

```bash
python3 src/client.py
```
