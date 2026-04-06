"""
Shared utilities: logging setup and JSON parsing.
"""

import json
import logging
import re
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def parse_json_response(raw: str) -> Any:
    """
    Parse a JSON string that may be wrapped in markdown code fences.

    Handles:
      - Plain JSON
      - ```json ... ```
      - ``` ... ```

    Raises:
        json.JSONDecodeError: if the content is not valid JSON after stripping.
    """
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)
