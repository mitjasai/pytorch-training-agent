# Agentic PyTorch hyperparameter tuning on LUMI

This repository contains an example of using an LLM agent to optimize the hyperparameters of a
PyTorch training job on LUMI. The naive approach taken by the agent may not be very useful in
practice, so it should rather be used as a starting point for designing custom agentic workflows.

The agent optimizes training hyperparameters using the so-called "grad student descent" scheme,
i.e., by choosing the hyperparameters manually and in an ad-hoc manner. For a description of this
approach, see, e.g., Section 2 of [Gencoglu et al. (2019)](https://arxiv.org/pdf/1904.07633). The
agent runs in the context of one GPU job, where it completes $n$ consecutive training runs,
updating the hyperparameters after each run, and finally reports the optimal configuration.

The agent (`src/agent.py`) is implemented using the
[LangChain](docs.langchain.com/oss/python/langchain/overview) framework. It trains models by
calling the `main` function of the training script (`src/train.py`) through an MCP server
(`src/mcp_server.py`). The training script used in this example is adapted (with minimal changes)
from the
[PyTorch quickstart tutorial](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).

## Usage

Copy the included `.env.example` to `.env` and replace the example values in the latter file with
those corresponding to your endpoint and model.

```bash
cp .env.example .env
```

The default values assume you are serving `google/gemma-4-31b-it` through an unauthenticated vLLM
endpoint. If you want to go the vLLM route, you can use the CSC-provided example script for
[running vLLM on LUMI](https://github.com/CSCfi/ai-inference-examples/blob/master/run-vllm-lumi4.sh).

Run the agent by submitting `run_agent.sh` to the Slurm queue.

```bash
sbatch -A project_xxxxxxxxx run_agent.sh
```

## Example output

```text
Based on the three training runs, the optimal hyperparameters among those tested
are:

**Optimal Hyperparameters:**
- **Batch Size:** 32
- **Learning Rate:** 0.01
- **Epochs:** 10

**Analysis of Training Runs:**
1. **Run 1 (`batch_size=32, lr=0.01, epochs=10`):** This run performed the best,
   achieving a final accuracy of **85.9%** and an average loss of **0.392**. The
   loss decreased steadily, and the accuracy increased consistently.
2. **Run 2 (`batch_size=64, lr=0.001, epochs=10`):** This run performed
   significantly worse, finishing with an accuracy of **70.0%** and a much
   higher average loss of **0.791**. This suggests that a lower learning rate
   and larger batch size slowed down convergence too much for the given number
   of epochs.
3. **Run 3 (`batch_size=32, lr=0.005, epochs=15`):** Despite more epochs, this
   run achieved a final accuracy of **85.4%** and an average loss of **0.410**.
   While competitive, it did not surpass the result of Run 1, indicating that
   the higher learning rate of 0.01 was more effective for this model.
```
