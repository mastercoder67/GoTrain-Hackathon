"""Optional stdio MCP adapter for GoPilot. Final test evaluation is intentionally excluded."""
import json
import os

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from .config import Recipe
from .store import best_run, list_runs

mcp = MCPServer("GoTrain")


@mcp.tool()
def inspect_dataset(task: str) -> dict:
    """Inspect a supported GEO-Bench task using only train and valid splits."""
    from .data import inspect
    return inspect(task)


@mcp.tool()
def train_model(task: str, recipe_json: str, hypothesis: str) -> dict:
    """Run one bounded GPU training experiment and record validation metrics."""
    from .engine import train
    return train(task, Recipe(**json.loads(recipe_json)), hypothesis)


@mcp.tool()
def diagnose_model(task: str, run_id: str = "") -> dict:
    """Measure where a completed run fails on validation data and suggest recipe levers."""
    from .diagnose import diagnose
    return diagnose(task, run_id or None)


@mcp.tool()
def recall_lessons(task: str) -> dict:
    """GoTrain's cross-task memory: measured lessons, a lever scoreboard and similar tasks' best recipes."""
    from .lessons import recall
    return recall(task)


@mcp.tool()
def compare_models(task: str) -> dict:
    """Return development results, with the model chosen by validation only."""
    return {"runs": list_runs(task), "best": best_run(task)}


@mcp.tool()
def upload_model(task: str, run_id: str) -> dict:
    """Publish a model and its checksummed manifest to filesystem and optional S3."""
    from .artifacts import publish_model
    return publish_model(task, run_id, os.environ.get("GOTRAIN_S3_BUCKET"))


@mcp.tool()
def download_model(task: str, run_id: str) -> dict:
    """Restore a published model and verify checksums."""
    from .artifacts import restore_model
    return restore_model(task, run_id, os.environ.get("GOTRAIN_S3_BUCKET"))


if __name__ == "__main__":
    mcp.run()
