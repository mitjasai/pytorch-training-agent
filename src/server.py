import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from fastmcp import FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

mcp = FastMCP("Number Guesser")


@mcp.tool
async def generate_number(
    low: int = 1, high: int = 100, ctx: Context = CurrentContext(),
) -> str:
    """
    Generate a secret number.

    Returns a string indicating successful generation.
    """
    rng = np.random.default_rng()
    rints = rng.integers(low=low, high=high, size=1)

    await ctx.set_state("secret_number", int(rints[0]))

    secret_number = await ctx.get_state("secret_number")
    logger.info(f"The secret number is {secret_number}.")

    return "Secret number generated."


@mcp.tool
async def guess_number(guess: int, ctx: Context = CurrentContext()) -> str:
    """
    Guess the secret number.

    Returns a message indicating whether the guess is correct,
    or if the secret number is lesser or greater than the guess.
    """
    secret_number = await ctx.get_state("secret_number")

    if guess < secret_number:
        return f"The secret number is greater than {guess}."
    elif guess == secret_number:
        return f"Correct, the secret number is {guess}!"
    elif guess > secret_number:
        return f"The secret number is lesser than {guess}."


if __name__ == "__main__":
    mcp.run()
