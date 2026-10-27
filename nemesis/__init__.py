"""Nemesis: the chess engine that plays the move that beats you."""

import os

# Nothing about your games leaves your computer. ONNX Runtime (which runs the Maia-2
# human-move model) sends usage telemetry to Microsoft from the moment it's imported
# unless this is set first, so it's set here, before anything can import it.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
