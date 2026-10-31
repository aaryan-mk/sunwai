"""Process-wide values set once by the entry point and read by stages."""

args_dur: float | None = None
"""Requested chunk duration in seconds (slot force-cut length); ``None`` when unknown."""
