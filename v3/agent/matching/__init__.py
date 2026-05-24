"""Candidate ranker and matcher.

Ranks cross-user candidates and manages sequential reveal.
"""
from .cross_user_retrieval import find_matches, get_sequential_candidate

__all__ = ["find_matches", "get_sequential_candidate"]
