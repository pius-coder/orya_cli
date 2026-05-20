"""Entity Tree management: routing, attachment, audit, propagation.

Simplified MemBrain-style entity tree in pure Python.
Each entity has a tree: root -> aspects -> leaves (facts).
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Hyperparameters (from MemBrain)
ALPHA_DESC = 0.5
TREE_MAX_CHILDREN = 15
AUDIT_MAX_K = 5
AUDIT_MIN_UNCERTAINTY = 0.5
MIN_FRESH_FOR_PROPAGATE = 3


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class TreeNode:
    node_type: str = "aspect"  # 'root' | 'aspect' | 'leaf'
    parent: Optional["TreeNode"] = None
    children: list["TreeNode"] = field(default_factory=list)
    fact_id: Optional[int] = None
    fact_text: Optional[str] = None
    fact_embedding: Optional[np.ndarray] = None
    description: Optional[str] = None
    description_embedding: Optional[np.ndarray] = None
    dirty: bool = False
    _removed: bool = False
    support: int = 0
    fresh_count: int = 0
    subtree_centroid: Optional[np.ndarray] = None


class EntityTree:
    def __init__(self, entity_id: str, root: TreeNode):
        self.entity_id = entity_id
        self.root = root
        self.leaf_index: dict[int, TreeNode] = {}  # fact_id -> leaf

    def all_nodes(self) -> list[TreeNode]:
        result = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node._removed:
                result.append(node)
                stack.extend(node.children)
        return result


def route_facts(tree: EntityTree, new_facts: list[dict[str, Any]]) -> dict[int, list[TreeNode]]:
    """Route new facts to the best attachment points in the tree.

    Returns: mapping of fact_id -> list of candidate parent nodes.
    """
    result: dict[int, list[TreeNode]] = {}
    for fact in new_facts:
        fid = fact.get("id")
        if fid is None:
            continue
        emb = fact.get("embedding")
        if emb is None:
            # Attach directly to root if no embedding
            result[fid] = [tree.root]
            continue

        # Top-down descent: score children by similarity
        current = tree.root
        path = [current]
        while current.children:
            # Score each child
            scored = []
            for child in current.children:
                if child._removed:
                    continue
                if child.node_type == "leaf":
                    sim = cosine_sim(emb, child.fact_embedding)
                else:
                    sim = ALPHA_DESC * cosine_sim(emb, child.description_embedding) + (
                        1 - ALPHA_DESC
                    ) * cosine_sim(emb, child.subtree_centroid)
                scored.append((sim, child))
            if not scored:
                break
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            path.append(best)
            current = best
            if current.node_type == "leaf":
                break
        result[fid] = path
    return result


def attach_all(
    tree: EntityTree,
    routing_result: dict[int, list[TreeNode]],
    facts: dict[int, dict[str, Any]],
) -> None:
    """Attach routed facts as new leaves."""
    for fid, path in routing_result.items():
        parent = path[-1]
        fact = facts.get(fid, {})
        leaf = TreeNode(
            node_type="leaf",
            parent=parent,
            fact_id=fid,
            fact_text=fact.get("text", ""),
            fact_embedding=fact.get("embedding"),
        )
        parent.children.append(leaf)
        tree.leaf_index[fid] = leaf
        # Update ancestors
        node = parent
        while node:
            node.support += 1
            node.fresh_count += 1
            node.dirty = True
            node = node.parent


def compute_debt(node: TreeNode) -> float:
    """Compute structural debt for audit prioritization."""
    active = [c for c in node.children if not c._removed]
    width_penalty = max(0, len(active) - 8)
    depth = 0
    n = node
    while n.parent:
        depth += 1
        n = n.parent
    depth_penalty = max(0, depth - 4)
    return (node.fresh_count * 0.1) + width_penalty * 0.5 + depth_penalty * 0.3


def auto_dissolve(tree: EntityTree) -> int:
    """Remove empty or single-child aspects."""
    removed = 0
    for node in tree.all_nodes():
        if node.node_type != "aspect":
            continue
        active = [c for c in node.children if not c._removed]
        if len(active) <= 1:
            node._removed = True
            removed += 1
    return removed


def batch_recompute_centroids(tree: EntityTree) -> None:
    """Recompute subtree centroids bottom-up."""
    # Process leaves first
    for node in tree.all_nodes():
        if node.node_type == "leaf" and node.fact_embedding is not None:
            node.subtree_centroid = node.fact_embedding.copy()
            node.support = 1
        elif node.node_type == "leaf":
            node.support = 1

    # Bottom-up pass (naive: just iterate multiple times)
    for _ in range(10):
        changed = False
        for node in tree.all_nodes():
            if node.node_type == "leaf":
                continue
            active = [c for c in node.children if not c._removed]
            if not active:
                continue
            embs = []
            for c in active:
                if c.subtree_centroid is not None:
                    embs.append(c.subtree_centroid * c.support)
            if embs:
                new_centroid = np.sum(embs, axis=0) / sum(c.support for c in active)
                if node.subtree_centroid is None or not np.allclose(
                    node.subtree_centroid, new_centroid, atol=1e-4
                ):
                    node.subtree_centroid = new_centroid
                    changed = True
        if not changed:
            break


def build_aspect_paths(tree: EntityTree) -> dict[int, str]:
    """Build human-readable aspect paths for each fact_id."""
    paths: dict[int, str] = {}
    for node in tree.all_nodes():
        if node.node_type != "leaf" or node.fact_id is None:
            continue
        parts = []
        n = node.parent
        while n and n != tree.root:
            if n.description:
                parts.append(n.description)
            n = n.parent
        parts.reverse()
        paths[node.fact_id] = " > ".join(parts) if parts else "General"
    return paths
