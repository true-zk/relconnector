"""Atomic-route RelGNN adapted from the authors' MIT-licensed implementation.

Source: https://github.com/snap-stanford/RelGNN (commit cffdb8b54627e92c7dd112c1243dde739c90d35b).
The existing online feature encoders, temporal encoding, and task heads remain
in the baseline/v2 packages; this module supplies only composite message passing.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor
from torch_geometric.nn import LayerNorm, SAGEConv, TransformerConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType

AtomicRoute = tuple[str, ...]


def atomic_routes(edge_types: list[EdgeType]) -> list[AtomicRoute]:
    """Use the authors' direct and fact-mediated atomic route construction."""
    from_source: dict[str, list[EdgeType]] = defaultdict(list)
    for source, relation, destination in edge_types:
        if relation.startswith("f2p_"):
            # A self FK is a distinct relation, even when its endpoints agree.
            from_source[source].append((source, relation, destination))

    routes: list[AtomicRoute] = []
    for relations in from_source.values():
        if len(relations) == 1:
            source, relation, destination = relations[0]
            routes.append(("dim-dim", source, relation, destination))
            routes.append(
                ("dim-dim", destination, f"rev_{relation}", source)
            )
            continue
        for source, relation_q, destination_q in relations:
            for _, relation_v, destination_v in relations:
                if relation_q == relation_v:
                    continue
                routes.append(
                    (
                        "dim-fact-dim",
                        source,
                        relation_q,
                        destination_q,
                        destination_v,
                        f"rev_{relation_v}",
                        source,
                    )
                )
    return routes


class _RouteConv(TransformerConv):
    def __init__(self, route_kind: str, channels: int, heads: int, aggr: str):
        super().__init__((channels, channels), channels, heads=heads)
        self.route_kind = route_kind
        self.aggr_conv = (
            SAGEConv((channels, channels), channels, aggr=aggr)
            if route_kind == "dim-fact-dim"
            else None
        )
        self.final_proj = Linear(heads * channels, channels)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "aggr_conv") and self.aggr_conv is not None:
            self.aggr_conv.reset_parameters()
        if hasattr(self, "final_proj"):
            self.final_proj.reset_parameters()

    def forward_direct(
        self, source: Tensor, destination: Tensor, edges: Tensor
    ) -> Tensor | None:
        if edges.shape[1] == 0:
            return None
        return self.final_proj(super().forward((source, destination), edges))

    def forward_composite(
        self,
        source: Tensor,
        middle: Tensor,
        destination: Tensor,
        aggregate_edges: Tensor,
        attention_edges: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        if attention_edges.shape[1] == 0:
            return None
        assert self.aggr_conv is not None
        message = (
            middle
            if aggregate_edges.shape[1] == 0
            else self.aggr_conv((source, middle), aggregate_edges)
        )
        output = super().forward((message, destination), attention_edges)
        return self.final_proj(output), message


class _CompositeLayer(torch.nn.Module):
    def __init__(
        self,
        node_types: list[NodeType],
        routes: list[AtomicRoute],
        channels: int,
        heads: int,
        aggr: str,
    ) -> None:
        super().__init__()
        self.routes = routes
        self.convs = ModuleDict(
            {route: _RouteConv(route[0], channels, heads, aggr) for route in routes}
        )
        self.norms = torch.nn.ModuleDict(
            {node_type: LayerNorm(channels, mode="node") for node_type in node_types}
        )
        self.aggr = aggr

    def reset_parameters(self) -> None:
        for conv in self.convs.values():
            conv.reset_parameters()
        for norm in self.norms.values():
            norm.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        outputs: dict[NodeType, list[Tensor]] = defaultdict(list)
        for route, conv in self.convs.items():
            if route[0] == "dim-dim":
                _, source, relation, destination = route
                edge = edge_index_dict.get((source, relation, destination))
                if edge is None or source not in x_dict or destination not in x_dict:
                    continue
                result = conv.forward_direct(x_dict[source], x_dict[destination], edge)
                if result is not None:
                    outputs[destination].append(result)
            else:
                _, middle, attention_rel, destination, source, aggregate_rel, _ = route
                attention = edge_index_dict.get((middle, attention_rel, destination))
                aggregate = edge_index_dict.get((source, aggregate_rel, middle))
                if (
                    attention is None
                    or aggregate is None
                    or source not in x_dict
                    or middle not in x_dict
                    or destination not in x_dict
                ):
                    continue
                result = conv.forward_composite(
                    x_dict[source], x_dict[middle], x_dict[destination],
                    aggregate, attention,
                )
                if result is not None:
                    dst_output, middle_output = result
                    outputs[destination].append(dst_output)
                    outputs[middle].append(middle_output)

        updated = dict(x_dict)
        for node_type, values in outputs.items():
            if self.aggr not in {"sum", "mean"}:
                raise ValueError(f"unsupported RelGNN route aggregation: {self.aggr}")
            # torch.stack(values) copies every route output into one more
            # [routes, nodes, channels] tensor. Salt has 20 routes and about
            # 245k fact nodes per batch: that temporary alone exceeds 2 GiB.
            combined = sum(values[1:], values[0])
            if self.aggr == "mean":
                combined = combined / len(values)
            updated[node_type] = combined
        return {
            node_type: self.norms[node_type](value).relu()
            for node_type, value in updated.items()
        }


class RelGNNBackbone(torch.nn.Module):
    """RelGNN composite message passing on the existing sampled FK graph."""

    def __init__(
        self,
        node_types: list[NodeType],
        edge_types: list[EdgeType],
        channels: int,
        num_layers: int,
        aggr: str,
        heads: int = 1,
    ) -> None:
        super().__init__()
        routes = atomic_routes(edge_types)
        if not routes:
            raise ValueError("RelGNN requires at least one atomic route")
        self.layers = torch.nn.ModuleList(
            _CompositeLayer(node_types, routes, channels, heads, aggr)
            for _ in range(num_layers)
        )
        self.routes = routes
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict
