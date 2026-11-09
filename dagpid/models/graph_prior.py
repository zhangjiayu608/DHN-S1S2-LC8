from __future__ import annotations

import torch


DEFAULT_GRAPH_NODES = (
    "sar",
    "optical",
    "joint",
    "dinov3",
    "local",
    "cf_sar",
    "cf_optical",
    "conflict",
    "boundary",
)

def build_graph_prior(
    nodes: tuple[str, ...] = DEFAULT_GRAPH_NODES,
    edges: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
) -> torch.Tensor:
    """Return soft prior edge targets; rows are senders and columns are receivers."""

    index = {name: i for i, name in enumerate(nodes)}
    prior = torch.zeros(len(nodes), len(nodes), dtype=torch.float32)
    if edges is None:
        edges = (
            ("sar", "joint"),
            ("optical", "joint"),
            ("optical", "dinov3"),
            ("local", "boundary"),
            ("sar", "cf_optical"),
            ("optical", "cf_sar"),
            ("dinov3", "cf_sar"),
            ("sar", "conflict"),
            ("optical", "conflict"),
            ("joint", "conflict"),
            ("dinov3", "conflict"),
            ("cf_sar", "conflict"),
            ("cf_optical", "conflict"),
            ("conflict", "boundary"),
        )
    for src, dst in edges:
        if src in index and dst in index and src != dst:
            prior[index[src], index[dst]] = 1.0
    return prior
