# LangChain agent example

This repository contains a very simple example of implementing an LLM agent. The example agent
calls the `guess_number` tool until it guesses correctly the randomly generated secret number.
While the script may not be especially useful in itself, it can be easily adapted for more
meaningful tasks by providing it with another tool.

## Usage

Copy the included `.env.example` to `.env` and replace the dummy values in the latter file with
real ones.

```bash
cp .env.example .env
```

Create a Python virtual environment and install the project.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the example script.

```bash
python3 example.py
```
