"""Training utilities for SLWM sprint smoke and synthetic-signal runs.

The namespace avoids importing runner modules eagerly so model modules can import
standalone loss helpers without creating circular imports.
"""

__all__: list[str] = []
