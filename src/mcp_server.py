from contextlib import redirect_stdout, redirect_stderr
from mcp.server.fastmcp import FastMCP
from pathlib import Path

from train import main as trainer

LOG_DIR = Path("logs")

mcp = FastMCP("PyTorch Trainer")


@mcp.tool()
def train_model(batch_size: int = 64, lr: float = 1e-3, epochs: int = 5) -> str:
    """
    Train a PyTorch image recognition model.

    Args:
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of training epochs

    Returns:
        Path to log file of completed training run
    """
    LOG_DIR.mkdir(exist_ok=True)

    log_file = LOG_DIR / f"batch_size_{batch_size}_lr_{lr}_epochs_{epochs}.log"

    with open(log_file, mode="w", buffering=1) as f:
        with redirect_stdout(f), redirect_stderr(f):
            trainer(batch_size, lr, epochs)

    return log_file


@mcp.tool()
def list_logs() -> list:
    """
    List log files of ongoing and completed training runs.

    Returns:
        List of log files
    """
    return [str(p) for p in LOG_DIR.glob("*")]


@mcp.tool()
def inspect_log(log_file) -> str:
    """
    Read the contents of a log file.

    Args:
        log_file: Path to log file

    Returns:
        Contents of log file
    """
    with open(log_file) as file:
        return file.read()


if __name__ == "__main__":
    mcp.run()
