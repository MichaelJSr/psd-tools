"""Resource limits shared by nested low-level PSD parsers."""

import operator
import sys
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any


class ParseLimitError(ValueError):
    """A structural parsing resource limit was exceeded."""


@dataclass(frozen=True)
class ParseLimits:
    """Limit reads, materialized bytes, parsed objects, and container nesting."""

    max_read_bytes: int | None = None
    max_total_bytes: int | None = None
    max_objects: int | None = 1_000_000
    max_nesting_depth: int | None = 64

    def __post_init__(self) -> None:
        for name in (
            "max_read_bytes",
            "max_total_bytes",
            "max_objects",
            "max_nesting_depth",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a positive integer or None")
            try:
                normalized = operator.index(value)
            except TypeError:
                raise TypeError(f"{name} must be a positive integer or None") from None
            if normalized <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, normalized)


class _ParseBudget:
    def __init__(self, limits: ParseLimits):
        self.limits = limits
        self.total_bytes = 0
        self.objects = 0
        self.depth = 0
        self.error: ParseLimitError | None = None
        self._max_read = limits.max_read_bytes
        self._max_total = limits.max_total_bytes
        self._max_objects = limits.max_objects

    def check(self, value: int, maximum: int | None, name: str) -> None:
        if self.error is not None:
            raise self.error
        if maximum is not None and value > maximum:
            self.error = ParseLimitError(
                "%s exceeded: %d > %d" % (name, value, maximum)
            )
            raise self.error

    def consume(self, objects: int, size: int) -> None:
        """``consume_objects(objects)``, then ``consume_bytes(size)``."""
        count = self.objects + objects
        total = self.total_bytes + size
        if (
            self.error is not None
            or (self._max_objects is not None and count > self._max_objects)
            or (self._max_read is not None and size > self._max_read)
            or (self._max_total is not None and total > self._max_total)
        ):
            # One of these raises.
            self.check(count, self.limits.max_objects, "max_objects")
            self.objects = count
            self.check(size, self.limits.max_read_bytes, "max_read_bytes")
            self.check(total, self.limits.max_total_bytes, "max_total_bytes")
        self.objects = count
        self.total_bytes = total


_budget: ContextVar[_ParseBudget | None] = ContextVar("psd_parse_budget", default=None)
current_budget = _budget.get


class _Context:
    """``parse_context()``; a class, as the generator form costs more per use."""

    __slots__ = ("_limits", "_budget", "_token")

    def __init__(self, limits: ParseLimits | None):
        self._limits = limits
        self._token: Token[_ParseBudget | None] | None = None

    def __enter__(self) -> bool:
        limits = self._limits
        if limits is not None and not isinstance(limits, ParseLimits):
            raise TypeError("parse_limits must be ParseLimits or None")
        if _budget.get() is not None:
            return False
        self._budget = _ParseBudget(limits if limits is not None else ParseLimits())
        self._token = _budget.set(self._budget)
        return True

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        token = self._token
        if token is not None:
            self._token = None
            try:
                if kind is None and self._budget.error is not None:
                    raise self._budget.error
            finally:
                _budget.reset(token)


def parse_context(limits: ParseLimits | None = None) -> AbstractContextManager[bool]:
    """Start a parse budget or reuse the current synchronous parse budget."""
    return _Context(limits)


def consume_bytes(size: int) -> None:
    """Reserve materialized bytes before reading or copying input."""
    budget = _budget.get()
    if budget is not None:
        budget.consume(0, size)


def consume_objects(count: int) -> None:
    """Reserve parsed elements, scalar values, or array entries."""
    budget = _budget.get()
    if budget is not None:
        budget.consume(count, 0)


def nesting_room() -> int | None:
    """Container levels left under the current budget; None without one or after a breach."""
    budget = _budget.get()
    if budget is None or budget.error is not None:
        return None
    maximum = budget.limits.max_nesting_depth
    return sys.maxsize if maximum is None else maximum - budget.depth


def charge(total_bytes: int, objects: int, largest_read: int) -> bool:
    """Charge a finished parse at once, or nothing and False when that breaks a limit.

    Equal to charging its reads one by one: the totals only grow, so the last
    sums are the largest.
    """
    budget = _budget.get()
    if budget is None or budget.error is not None:
        return False
    limits = budget.limits
    if (
        (limits.max_read_bytes is not None and largest_read > limits.max_read_bytes)
        or (
            limits.max_total_bytes is not None
            and budget.total_bytes + total_bytes > limits.max_total_bytes
        )
        or (
            limits.max_objects is not None
            and budget.objects + objects > limits.max_objects
        )
    ):
        return False
    budget.total_bytes += total_bytes
    budget.objects += objects
    return True


class _Container:
    __slots__ = ("_budget",)

    def __enter__(self) -> None:
        budget = _budget.get()
        if budget is None:
            raise RuntimeError("Container parsing requires a parse context")
        budget.check(
            budget.depth + 1, budget.limits.max_nesting_depth, "max_nesting_depth"
        )
        budget.depth += 1
        self._budget = budget

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        self._budget.depth -= 1


def parse_container() -> AbstractContextManager[None]:
    """Reserve one active descriptor or text-engine container level."""
    return _Container()
