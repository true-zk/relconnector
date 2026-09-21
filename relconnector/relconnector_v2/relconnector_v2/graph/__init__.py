"""In-memory graph topology used by online samplers."""

from .builder import InMemoryGraphIndexBuilder
from .index import GraphIndex

__all__ = ["GraphIndex", "InMemoryGraphIndexBuilder"]
