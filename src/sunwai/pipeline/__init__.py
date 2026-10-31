"""Dubbing pipeline: Stage interface, runner, per-stage adapters and the chunk workflow.

``engine`` is the original, battle-tested implementation of every stage; the ``stages`` package exposes each of its
steps behind the common :class:`~sunwai.pipeline.stage.Stage` interface.
"""
