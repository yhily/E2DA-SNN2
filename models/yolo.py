"""Backward-compatible import path for checkpoints released before the rename.

The public 2025 checkpoint serializes ``Model`` and ``Detect`` under
``models.yolo``.  The maintained implementation now lives in ``models.model``;
re-exporting the classes keeps the published weights loadable without changing
their tensors.
"""

from models.model import Detect, Model

__all__ = ["Detect", "Model"]
