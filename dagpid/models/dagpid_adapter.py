from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .adapters import SensorConditionedAdapter
from .graph_prior import DEFAULT_GRAPH_NODES, build_graph_prior
from .decoder import DepthwiseSeparableConvModule, TokenPyramidUPerDecoder
from .preprocessing import normalize_local
from .pid_control import PIDBagFusion


def _zero_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SequenceSSMBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        inner_dim: int | None = None,
        mlp_ratio: float = 1.5,
        layer_scale: float = 0.01,
    ) -> None:
        super().__init__()
        hidden = max(dim, int(dim * float(mlp_ratio)))
        self.scan = self._build_scan(dim=dim, inner_dim=inner_dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.scan_scale = nn.Parameter(torch.full((dim,), float(layer_scale)))
        self.ffn_scale = nn.Parameter(torch.full((dim,), float(layer_scale)))
        _zero_init(self.ffn[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scan_scale.to(dtype=x.dtype) * self.scan(x)
        x = x + self.ffn_scale.to(dtype=x.dtype) * self.ffn(self.norm(x))
        return x

    @staticmethod
    def _build_scan(dim: int, inner_dim: int | None) -> nn.Module:
        try:
            try:
                from mamba_ssm import Mamba
            except Exception:
                from mamba_ssm.modules.mamba_simple import Mamba
            return Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        except Exception as error:
            raise RuntimeError(
                "Mamba scan backend was requested but could not be constructed"
            ) from error


class LiteVSSBlock2D(nn.Module):
    """Two-axis lightweight visual selective scan for 12x12 token maps."""

    def __init__(
        self,
        dim: int,
        inner_dim: int | None = None,
        mlp_ratio: float = 1.5,
        layer_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.scan_hw = SequenceSSMBlock(
            dim=dim, inner_dim=inner_dim, mlp_ratio=mlp_ratio, layer_scale=layer_scale
        )
        self.scan_wh = SequenceSSMBlock(
            dim=dim, inner_dim=inner_dim, mlp_ratio=mlp_ratio, layer_scale=layer_scale
        )
        self.mix = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        _zero_init(self.mix[-1])
        self.mix_scale = nn.Parameter(torch.full((1, dim, 1, 1), float(layer_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        seq_hw = x.flatten(2).transpose(1, 2)
        out_hw = (
            self.scan_hw(seq_hw).transpose(1, 2).reshape(bsz, channels, height, width)
        )
        seq_wh = x.transpose(2, 3).flatten(2).transpose(1, 2)
        out_wh = (
            self.scan_wh(seq_wh)
            .transpose(1, 2)
            .reshape(bsz, channels, width, height)
            .transpose(2, 3)
        )
        delta = self.mix(torch.cat([out_hw, out_wh], dim=1))
        return x + self.mix_scale.to(dtype=x.dtype) * delta


class LocalTokenEncoder(nn.Module):
    """Lightweight S1/S2 local evidence encoder for DAGPID-Adapter."""

    def __init__(
        self,
        in_channels: int = 14,
        out_dim: int = 768,
        stem_dim: int = 96,
        vss_dim: int = 192,
        depth: int = 2,
        patch_size: int = 8,
        ssm_inner_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(
                stem_dim,
                vss_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
            ),
            nn.BatchNorm2d(vss_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                LiteVSSBlock2D(dim=vss_dim, inner_dim=ssm_inner_dim or vss_dim)
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(vss_dim)
        self.proj = nn.Linear(vss_dim, out_dim)

    def forward(
        self, s1: torch.Tensor, s2: torch.Tensor, branch: str = "both"
    ) -> torch.Tensor:
        s1_in = self._normalize_local(s1)
        s2_in = self._normalize_local(s2)
        if branch == "sar":
            s2_in = torch.zeros_like(s2_in)
        elif branch == "optical":
            s1_in = torch.zeros_like(s1_in)
        fmap = self.stem(torch.cat([s1_in, s2_in], dim=1))
        fmap = self.blocks(fmap)
        tokens = fmap.flatten(2).transpose(1, 2)
        return self.proj(self.norm(tokens))

    @staticmethod
    def _normalize_local(x: torch.Tensor) -> torch.Tensor:
        return normalize_local(x)


class LiteTokenIntegralContext(nn.Module):

    def __init__(
        self, dim: int, bottleneck_dim: int = 192, scales: tuple[int, ...] = (1, 2, 3)
    ) -> None:
        super().__init__()
        self.scales = tuple((int(s) for s in scales))
        self.proj = nn.Sequential(
            nn.Conv2d(
                dim * (len(self.scales) + 1), bottleneck_dim, kernel_size=1, bias=False
            ),
            nn.GroupNorm(math.gcd(32, bottleneck_dim), bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(math.gcd(32, dim), dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor, target_side: int) -> torch.Tensor:
        bsz, count, dim = tokens.shape
        if target_side * target_side != count:
            raise ValueError(
                f"LiteTokenIntegralContext expects {target_side ** 2} tokens, got {count}"
            )
        fmap = tokens.transpose(1, 2).reshape(bsz, dim, target_side, target_side)
        pieces = [fmap]
        for scale in self.scales:
            pooled = F.adaptive_avg_pool2d(fmap, output_size=(scale, scale))
            pieces.append(
                F.interpolate(
                    pooled,
                    size=(target_side, target_side),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        out = self.proj(torch.cat(pieces, dim=1))
        return out.flatten(2).transpose(1, 2).contiguous()


class RegionalFeatureRefinement(nn.Module):
    """Shared cluster-centric SSM refinement over each evidence token stream."""

    def __init__(
        self,
        dim: int,
        modalities: tuple[str, ...],
        ssm_dim: int = 192,
        cluster_hw: int = 4,
        temperature: float = 0.08,
        depth: int = 1,
        refine_modalities: tuple[str, ...] | list[str] | str | None = None,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.refine_modalities = self._normalize_refine_modalities(refine_modalities)
        self.cluster_hw = int(cluster_hw)
        self.temperature = max(float(temperature), 0.0001)
        self.in_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ssm_dim))
        self.stream_prompt = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, 1, ssm_dim))
                for name in self.refine_modalities
            }
        )
        self.blocks = nn.Sequential(
            *[
                SequenceSSMBlock(dim=ssm_dim, inner_dim=ssm_dim)
                for _ in range(int(depth))
            ]
        )
        self.out_proj = nn.Linear(ssm_dim, dim)
        _zero_init(self.out_proj)

    def _normalize_refine_modalities(
        self, refine_modalities: tuple[str, ...] | list[str] | str | None
    ) -> tuple[str, ...]:
        if refine_modalities is None:
            return self.modalities
        if isinstance(refine_modalities, str):
            value = refine_modalities.strip()
            if value in {"", "*", "all"}:
                return self.modalities
            if value.lower() in {"none", "off", "false"}:
                return ()
            names = tuple((part.strip() for part in value.split(",") if part.strip()))
        else:
            names = tuple(
                (str(part).strip() for part in refine_modalities if str(part).strip())
            )
        valid = set(self.modalities)
        unknown = [name for name in names if name not in valid]
        if unknown:
            raise ValueError(
                f"Unknown refine_modalities for RegionalFeatureRefinement: {unknown}"
            )
        return names

    def forward(
        self, name: str, tokens: torch.Tensor, target_side: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, count, _ = tokens.shape
        if count != target_side * target_side:
            raise ValueError(
                f"RegionalFeatureRefinement expects {target_side ** 2} tokens, got {count}"
            )
        if name not in self.refine_modalities:
            return (tokens, {})
        low = self.in_proj(tokens)
        if name in self.stream_prompt:
            low = low + self.stream_prompt[name].to(dtype=low.dtype)
        fmap = low.transpose(1, 2).reshape(bsz, low.shape[-1], target_side, target_side)
        centers = F.adaptive_avg_pool2d(
            fmap, output_size=(self.cluster_hw, self.cluster_hw)
        )
        centers = centers.flatten(2).transpose(1, 2).contiguous()
        sim = torch.matmul(
            F.normalize(low.float(), dim=-1),
            F.normalize(centers.float(), dim=-1).transpose(1, 2),
        )
        assign = F.softmax(sim / self.temperature, dim=-1).to(dtype=low.dtype)
        denom = assign.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1e-06)
        aggregated = torch.matmul(assign.transpose(1, 2), low) / denom
        centers = centers + aggregated
        centers = self.blocks(centers)
        refined_low = torch.matmul(assign, centers)
        refined = tokens + self.out_proj(refined_low).to(dtype=tokens.dtype)
        return (refined, {})


class TaskStateEncoder(nn.Module):

    def __init__(
        self,
        dim: int,
        nodes: tuple[str, ...],
        evidence_nodes: tuple[str, ...],
        state_dim: int = 96,
    ) -> None:
        super().__init__()
        self.nodes = tuple(nodes)
        self.evidence_nodes = tuple(evidence_nodes)
        self.shared_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, state_dim))
        self.proxy_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, state_dim))
        self.missing = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, 1, state_dim))
                for name in self.evidence_nodes
            }
        )
        self.node_bias = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1, 1, state_dim)) for name in self.nodes}
        )

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        active_names: list[str],
        reference: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del active_names, source_ids
        bsz, token_count, dim = reference.shape
        states: dict[str, torch.Tensor] = {}
        for name in self.evidence_nodes:
            x = tokens.get(name)
            if x is None:
                state = (
                    self.missing[name]
                    .to(device=reference.device, dtype=reference.dtype)
                    .expand(bsz, token_count, -1)
                )
            else:
                state = self.shared_proj(x)
            states[name] = state + self.node_bias[name].to(dtype=state.dtype)

        def evidence(name: str) -> torch.Tensor:
            x = tokens.get(name)
            if x is None:
                return reference.new_zeros(bsz, token_count, dim)
            return x

        sar = evidence("sar")
        optical = evidence("optical")
        joint = evidence("joint")
        dinov3 = evidence("dinov3")
        cf_sar = evidence("cf_sar")
        cf_optical = evidence("cf_optical")
        local = evidence("local")
        conflict_terms = [(sar - optical).abs(), (joint - reference).abs()]
        if "cf_sar" in tokens:
            conflict_terms.append((cf_sar - sar).abs())
        if "cf_optical" in tokens:
            conflict_terms.append((cf_optical - optical).abs())
        if "dinov3" in tokens:
            conflict_terms.append((dinov3 - optical).abs())
        conflict = torch.stack(conflict_terms, dim=0).mean(dim=0)
        states["conflict"] = self.proxy_proj(conflict) + self.node_bias["conflict"].to(
            dtype=reference.dtype
        )
        local_grad = self._token_gradient(local if "local" in tokens else reference)
        boundary = 0.5 * local_grad + 0.5 * (local - reference).abs()
        states["boundary"] = self.proxy_proj(boundary) + self.node_bias["boundary"].to(
            dtype=reference.dtype
        )
        return {name: states[name] for name in self.nodes if name in states}

    @staticmethod
    def _token_gradient(tokens: torch.Tensor) -> torch.Tensor:
        bsz, count, dim = tokens.shape
        side = int(math.sqrt(count))
        if side * side != count or side <= 1:
            return torch.zeros_like(tokens)
        fmap = tokens.transpose(1, 2).reshape(bsz, dim, side, side)
        dx = F.pad((fmap[:, :, :, 1:] - fmap[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((fmap[:, :, 1:, :] - fmap[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        grad = 0.5 * (dx + dy)
        return grad.flatten(2).transpose(1, 2).contiguous()


class DirectedFeatureGraph(nn.Module):

    def __init__(
        self,
        nodes: tuple[str, ...],
        state_dim: int = 96,
        topology: tuple[str, ...] | list[str] | None = None,
        prior_edges: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
        blocked_edges: (
            list[tuple[str, str]] | tuple[tuple[str, str], ...] | None
        ) = None,
        delta_rank: int = 32,
        init_logit: float = -4.5,
        prior_logit: float = 1.6,
        delta_scale: float = 0.12,
    ) -> None:
        super().__init__()
        self.nodes = tuple(nodes)
        if len(set(self.nodes)) != len(self.nodes) or len(self.nodes) < 2:
            raise ValueError(
                "Graph nodes must be unique and include at least two nodes."
            )
        self.node_to_idx = {name: i for i, name in enumerate(self.nodes)}
        default_topology = (
            "sar",
            "optical",
            "dinov3",
            "local",
            "cf_sar",
            "cf_optical",
            "joint",
            "conflict",
            "boundary",
        )
        topology = tuple(topology or default_topology)
        if len(set(topology)) != len(topology):
            raise ValueError("Graph topology must not repeat nodes.")
        if (
            any((name not in self.node_to_idx for name in topology))
            and topology != default_topology
        ):
            raise ValueError("Graph topology contains unknown nodes.")
        ordered = [name for name in topology if name in self.node_to_idx]
        ordered.extend([name for name in self.nodes if name not in ordered])
        self.order_indices = [self.node_to_idx[name] for name in ordered]
        pos = {name: i for i, name in enumerate(ordered)}
        forward = torch.zeros(len(self.nodes), len(self.nodes), dtype=torch.float32)
        for src in self.nodes:
            for dst in self.nodes:
                if src != dst and pos[src] < pos[dst]:
                    forward[self.node_to_idx[src], self.node_to_idx[dst]] = 1.0
        for src, dst in blocked_edges or ():
            if src in self.node_to_idx and dst in self.node_to_idx:
                forward[self.node_to_idx[src], self.node_to_idx[dst]] = 0.0
        self.register_buffer("forward_mask", forward)
        prior = build_graph_prior(self.nodes, prior_edges) * forward
        self.register_buffer("prior", prior)
        logits = torch.full_like(prior, float(init_logit))
        logits = torch.where(
            prior > 0, torch.full_like(logits, float(prior_logit)), logits
        )
        logits = torch.where(forward > 0, logits, torch.full_like(logits, -12.0))
        logits.fill_diagonal_(-12.0)
        self.global_logits = nn.Parameter(logits)
        self.node_scan = SequenceSSMBlock(
            dim=state_dim, inner_dim=state_dim, mlp_ratio=1.5
        )
        self.adj_q = nn.Linear(state_dim, int(delta_rank), bias=False)
        self.adj_k = nn.Linear(state_dim, int(delta_rank), bias=False)
        self.delta_scale = float(delta_scale)

    def forward(self, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stacked = torch.stack([states[name] for name in self.nodes], dim=1)
        pooled = stacked.mean(dim=2)
        ordered = pooled[:, self.order_indices]
        ordered_context = self.node_scan(ordered)
        context_pooled = pooled.new_zeros(pooled.shape)
        context_pooled[:, self.order_indices] = ordered_context
        q = self.adj_q(context_pooled.float())
        k = self.adj_k(context_pooled.float())
        delta = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(max(q.shape[-1], 1))
        logits = self.global_logits.unsqueeze(
            0
        ).float() + self.delta_scale * torch.tanh(delta)
        mask = self.forward_mask.to(device=logits.device, dtype=logits.dtype).unsqueeze(
            0
        )
        logits = torch.where(mask > 0, logits, torch.full_like(logits, -12.0))
        adjacency = torch.sigmoid(logits) * mask
        incoming = torch.einsum(
            "bst,bsnd->btnd", adjacency.to(dtype=stacked.dtype), stacked
        )
        contexts = {name: incoming[:, i] for i, name in enumerate(self.nodes)}
        return {
            "adjacency": adjacency,
            "adjacency_mean": adjacency.mean(dim=0),
            "contexts": contexts,
            "prior": self.prior.to(device=adjacency.device, dtype=adjacency.dtype),
        }


class DAGPIDAdapter(nn.Module):
    """Regional evidence refinement, directed graph messages, and PID fusion."""

    def __init__(
        self,
        dim: int,
        modalities: tuple[str, ...] = (
            "sar",
            "optical",
            "joint",
            "dinov3",
            "local",
            "cf_sar",
            "cf_optical",
        ),
        rank: int = 64,
        reliability_hidden: int = 192,
        graph_context_dim: int = 32,
        uncertainty_gamma: float = 1.0,
        integral_gain: float = 0.35,
        derivative_gain: float = 0.78,
        feature_graph: dict | None = None,
        cluster_ssm: dict | None = None,
        graph_alpha_init: float = 0.04,
        graph_risk_beta_init: float = 0.08,
        graph_gate_scale_init: float = 0.18,
        initial_gate_bias: dict[str, float] | None = None,
        final_gate_bias: dict[str, float] | None = None,
        gate_warmup_epochs: int = 10,
        gate_warmup_start_epoch: int = 1,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.uncertainty_gamma = float(uncertainty_gamma)
        self.integral_gain = float(integral_gain)
        self.derivative_gain = float(derivative_gain)
        self.initial_gate_bias = {
            str(k): float(v) for k, v in (initial_gate_bias or {}).items()
        }
        self.final_gate_bias = {
            str(k): float(v) for k, v in (final_gate_bias or {}).items()
        }
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.gate_warmup_start_epoch = int(gate_warmup_start_epoch)
        self.current_epoch = 1
        self.adapters = nn.ModuleDict(
            {
                name: SensorConditionedAdapter(dim=dim, rank=rank)
                for name in self.modalities
            }
        )
        cluster_cfg = dict(cluster_ssm or {})
        self.evidence_ssm = RegionalFeatureRefinement(
            dim=dim, modalities=self.modalities, **cluster_cfg
        )
        graph_cfg = dict(feature_graph or {})
        graph_keys = {
            "nodes",
            "topology",
            "state_dim",
            "prior_edges",
            "blocked_edges",
            "delta_rank",
            "init_logit",
            "prior_logit",
            "delta_scale",
            "integral_bottleneck_dim",
            "type",
        }
        if set(graph_cfg) - graph_keys:
            raise ValueError(
                f"Unknown feature graph settings: {sorted(set(graph_cfg) - graph_keys)}"
            )
        extra_nodes = tuple(
            (name for name in self.modalities if name not in DEFAULT_GRAPH_NODES)
        )
        self.graph_nodes = tuple(
            graph_cfg.get("nodes", DEFAULT_GRAPH_NODES + extra_nodes)
        )
        proxy_nodes = {"conflict", "boundary"}
        evidence_nodes = tuple(
            (name for name in self.graph_nodes if name not in proxy_nodes)
        )
        state_dim = int(graph_cfg.get("state_dim", max(64, graph_context_dim * 2)))
        self.state_encoder = TaskStateEncoder(
            dim=dim,
            nodes=self.graph_nodes,
            evidence_nodes=evidence_nodes,
            state_dim=state_dim,
        )
        self.feature_graph = DirectedFeatureGraph(
            nodes=self.graph_nodes,
            state_dim=state_dim,
            topology=graph_cfg.get("topology"),
            prior_edges=graph_cfg.get("prior_edges"),
            blocked_edges=graph_cfg.get("blocked_edges"),
            delta_rank=int(graph_cfg.get("delta_rank", 32)),
            init_logit=float(graph_cfg.get("init_logit", -4.5)),
            prior_logit=float(graph_cfg.get("prior_logit", 1.6)),
            delta_scale=float(graph_cfg.get("delta_scale", 0.12)),
        )
        self.state_to_token = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, dim)
        )
        _zero_init(self.state_to_token[-1])
        self.integral_context = LiteTokenIntegralContext(
            dim=dim,
            bottleneck_dim=int(
                graph_cfg.get("integral_bottleneck_dim", max(128, reliability_hidden))
            ),
        )
        self.pag = nn.ModuleDict(
            {
                m: LiteTokenPagFM(dim=dim, mid_dim=max(32, dim // 8))
                for m in self.modalities
            }
        )
        rel_in_dim = dim + state_dim
        self.rel_head = nn.Sequential(
            nn.LayerNorm(rel_in_dim),
            nn.Linear(rel_in_dim, reliability_hidden),
            nn.GELU(),
            nn.Linear(reliability_hidden, 2),
        )
        self.i_head = nn.Sequential(
            nn.LayerNorm(rel_in_dim),
            nn.Linear(rel_in_dim, reliability_hidden),
            nn.GELU(),
            nn.Linear(reliability_hidden, 1),
        )
        d_in_dim = dim * 3 + state_dim
        self.d_head = nn.Sequential(
            nn.LayerNorm(d_in_dim),
            nn.Linear(d_in_dim, reliability_hidden),
            nn.GELU(),
            nn.Linear(reliability_hidden, 1),
        )
        self.graph_gate_head = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, 1)
        )
        self.graph_risk_head = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, 1)
        )
        self.stream_context = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, 1, state_dim))
                for name in self.modalities
            }
        )
        self.graph_alpha = nn.Parameter(
            torch.tensor(float(graph_alpha_init), dtype=torch.float32)
        )
        risk_beta_init = max(float(graph_risk_beta_init), 1e-06)
        self.graph_risk_beta = nn.Parameter(
            torch.tensor(math.log(math.expm1(risk_beta_init)), dtype=torch.float32)
        )
        self.graph_gate_scale = nn.Parameter(
            torch.tensor(float(graph_gate_scale_init), dtype=torch.float32)
        )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        present: dict[str, bool] | None = None,
        target_side: int | None = None,
        source_ids: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        present = present or {m: True for m in self.modalities}
        target_side = target_side or self._infer_target_side(tokens)
        active_names: list[str] = []
        refined: dict[str, torch.Tensor] = {}
        aux: dict[str, torch.Tensor] = {}
        for name in self.modalities:
            x = tokens.get(name)
            if x is None:
                continue
            x = self._align_tokens(x, target_side)
            assert self.evidence_ssm is not None
            adapted = self.adapters[name](x)
            refined[name], cluster_aux = self.evidence_ssm(name, adapted, target_side)
            aux.update(cluster_aux)
            active_names.append(name)
        if not active_names:
            raise RuntimeError("DAGPIDAdapter received no valid token streams")
        reference_names = [
            name for name in ("sar", "optical", "joint") if name in refined
        ]
        reference = (
            torch.stack([refined[name] for name in reference_names], dim=0)
            .mean(dim=0)
            .detach()
        )
        assert self.state_encoder is not None and self.feature_graph is not None
        states = self.state_encoder(
            refined, active_names, reference, source_ids=source_ids
        )
        graph = self.feature_graph(states)
        contexts = graph["contexts"]
        if return_aux:
            self._attach_graph_aux(aux, graph)
        alpha = torch.tanh(self.graph_alpha).to(dtype=reference.dtype)
        risk_beta = F.softplus(self.graph_risk_beta).to(dtype=reference.dtype)
        gate_scale = torch.tanh(self.graph_gate_scale).to(dtype=reference.dtype)
        controlled: dict[str, torch.Tensor] = {}
        gate_logits: list[torch.Tensor] = []
        risk_items: list[torch.Tensor] = []
        proxy_readout = torch.stack(
            [states[name] + contexts[name] for name in ("conflict", "boundary")], dim=0
        ).mean(dim=0)
        for name in active_names:
            context = contexts.get(name)
            if context is None:
                context = reference.new_zeros(
                    reference.shape[0],
                    reference.shape[1],
                    next(iter(contexts.values())).shape[-1],
                )
            context = context.to(dtype=refined[name].dtype)
            if name in self.stream_context:
                context = context + self.stream_context[name].to(dtype=context.dtype)
            context = 0.5 * (context + proxy_readout.to(dtype=context.dtype))
            message = self.state_to_token(context).to(dtype=refined[name].dtype)
            tokens_i = refined[name] + alpha * message
            integral = self.integral_context(tokens_i, target_side=target_side)
            controlled[name], sim = self.pag[name](tokens_i, integral)
            rel_input = torch.cat([tokens_i, context], dim=-1)
            rel_raw, unc_raw = self.rel_head(rel_input).split(1, dim=-1)
            unc = F.softplus(unc_raw) + 1e-06
            i_logit = self.i_head(torch.cat([integral, context], dim=-1))
            disagreement = (tokens_i - reference).abs()
            risk = (
                F.softplus(
                    self.d_head(
                        torch.cat([tokens_i, integral, disagreement, context], dim=-1)
                    )
                )
                + 1e-06
            )
            risk = risk + risk_beta * F.softplus(self.graph_risk_head(context)).to(
                dtype=risk.dtype
            )
            graph_bias = self.graph_gate_head(context).to(dtype=rel_raw.dtype)
            p_logit = rel_raw - self.uncertainty_gamma * unc
            logit = (
                p_logit
                + self.integral_gain * i_logit
                - self.derivative_gain * risk
                + gate_scale * graph_bias
                + self._gate_bias(name, rel_raw.device, rel_raw.dtype)
            )
            risk_items.append(risk)
            if not present.get(name, True):
                logit = torch.full_like(logit, -10000.0)
            gate_logits.append(logit)
        weights_tensor = torch.softmax(torch.stack(gate_logits, dim=0), dim=0)
        fused = None
        risk_token = None
        for i, name in enumerate(active_names):
            weight = weights_tensor[i]
            term = weight * controlled[name]
            fused = term if fused is None else fused + term
            risk_term = weight * risk_items[i]
            risk_token = risk_term if risk_token is None else risk_token + risk_term
        assert fused is not None
        assert risk_token is not None
        aux["pid_risk_token"] = risk_token
        if not return_aux:
            aux = {"pid_risk_token": risk_token}
        return (fused, aux)

    def _attach_graph_aux(
        self, aux: dict[str, torch.Tensor], graph: dict[str, torch.Tensor]
    ) -> None:
        adjacency = graph["adjacency_mean"]
        prior = graph["prior"]
        prior_mask = prior > 0
        allowed = self.feature_graph.forward_mask.to(dtype=adjacency.dtype)
        aux["graph_prior_loss"] = (
            (adjacency - prior).abs() * prior_mask
        ).sum() / prior_mask.sum().clamp_min(1)
        aux["graph_sparsity_loss"] = (
            adjacency * allowed
        ).sum() / allowed.sum().clamp_min(1)
        aux["graph_adjacency"] = adjacency

    def _gate_bias(
        self, name: str, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        final = self.final_gate_bias.get(name, 0.0)
        if name not in self.initial_gate_bias:
            return torch.tensor(final, device=device, dtype=dtype)
        start = self.gate_warmup_start_epoch
        if self.current_epoch < start:
            progress = 0.0
        elif self.gate_warmup_epochs <= 1:
            progress = 1.0
        else:
            progress = min(
                1.0,
                max(
                    0.0,
                    (self.current_epoch - start + 1) / float(self.gate_warmup_epochs),
                ),
            )
        initial = self.initial_gate_bias[name]
        return torch.tensor(
            initial + (final - initial) * progress, device=device, dtype=dtype
        )

    @staticmethod
    def _align_tokens(tokens: torch.Tensor, target_side: int) -> torch.Tensor:
        bsz, count, dim = tokens.shape
        if count == target_side * target_side:
            return tokens
        side = int(math.sqrt(count))
        if side * side != count:
            if count == 1:
                return tokens.expand(-1, target_side * target_side, -1).contiguous()
            raise ValueError(f"Token count must be square or 1, got {count}")
        fmap = tokens.transpose(1, 2).reshape(bsz, dim, side, side)
        fmap = F.interpolate(
            fmap, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        return fmap.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _infer_target_side(tokens: dict[str, torch.Tensor]) -> int:
        for key in (
            "joint",
            "optical",
            "sar",
            "local",
            "dinov3",
            "cf_sar",
            "cf_optical",
            "remoteclip",
        ):
            x = tokens.get(key)
            if x is None:
                continue
            side = int(math.sqrt(x.shape[1]))
            if side * side == x.shape[1]:
                return side
        raise ValueError("Cannot infer token grid side from non-square streams")


class LiteTokenPagFM(nn.Module):

    def __init__(self, dim: int, mid_dim: int = 96) -> None:
        super().__init__()
        self.f_x = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mid_dim))
        self.f_y = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mid_dim))
        self.scale = mid_dim ** (-0.5)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        similarity = (self.f_x(x).float() * self.f_y(y).float()).sum(
            dim=-1, keepdim=True
        ) * self.scale
        sim = torch.sigmoid(similarity).to(dtype=x.dtype)
        return ((1.0 - sim) * x + sim * y, sim)


class BoundaryAwareDecoder(nn.Module):
    """PID decoder with a low-resolution VSS risk mixer."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        channels: int = 192,
        local_in_channels: int = 14,
        use_local_stem: bool = True,
        risk_dim: int = 96,
        risk_downsample: int = 4,
    ) -> None:
        super().__init__()
        self.core = TokenPyramidUPerDecoder(
            in_dim=in_dim, num_classes=num_classes, channels=channels, feature_only=True
        )
        self.use_local_stem = bool(use_local_stem)
        self.local_stem = (
            nn.Sequential(
                DepthwiseSeparableConvModule(local_in_channels, channels // 2, 3, 1),
                DepthwiseSeparableConvModule(channels // 2, channels, 3, 1),
            )
            if self.use_local_stem
            else None
        )
        self.risk_downsample = max(1, int(risk_downsample))
        self.risk_reduce = nn.Sequential(
            nn.Conv2d(channels, risk_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(risk_dim),
            nn.GELU(),
        )
        self.risk_vss = LiteVSSBlock2D(dim=risk_dim, inner_dim=risk_dim)
        self.risk_head = nn.Conv2d(risk_dim, 1, kernel_size=1)
        self.pid_risk_scale = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))
        self.pid_bag = PIDBagFusion(channels=channels, risk_channels=1)
        self.branch_bias = nn.Parameter(torch.zeros(3, channels, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(channels, num_classes, kernel_size=1)
        )

    def forward(
        self,
        token_map: torch.Tensor,
        s1: torch.Tensor | None = None,
        s2: torch.Tensor | None = None,
        branch: str = "both",
        pid_risk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        integral_feature = self.core.forward_feature(token_map)
        if self.local_stem is not None and s1 is not None and (s2 is not None):
            local = self._local_input(s1, s2, branch=branch)
            proportional_feature = self.local_stem(
                local.to(dtype=integral_feature.dtype)
            )
        else:
            proportional_feature = torch.zeros_like(integral_feature)
        diff = (proportional_feature - integral_feature).abs()
        risk_base = diff
        if self.risk_downsample > 1:
            risk_base = F.avg_pool2d(
                risk_base, kernel_size=self.risk_downsample, stride=self.risk_downsample
            )
        risk_low = self.risk_vss(self.risk_reduce(risk_base))
        derivative_logits = self.risk_head(risk_low)
        derivative_logits = F.interpolate(
            derivative_logits,
            size=integral_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if pid_risk is not None:
            risk = F.interpolate(
                pid_risk.to(dtype=derivative_logits.dtype),
                size=derivative_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            derivative_logits = (
                derivative_logits
                + self.pid_risk_scale.to(dtype=derivative_logits.dtype) * risk
            )
        fused, risk_attention = self.pid_bag(
            proportional_feature, integral_feature, derivative_logits
        )
        fused = fused + self.branch_bias[self._branch_index(branch)].to(
            dtype=fused.dtype
        )
        logits = self.classifier(fused)
        aux: dict[str, torch.Tensor] = {
            "decoder_feature": fused,
            "pid_boundary_logits": derivative_logits,
            "pid_decoder_risk_attention": risk_attention.mean(dim=1, keepdim=True),
        }
        return (logits, aux)

    @staticmethod
    def _branch_index(branch: str) -> int:
        if branch == "sar":
            return 1
        if branch == "optical":
            return 2
        return 0

    @staticmethod
    def _normalize_local(x: torch.Tensor) -> torch.Tensor:
        return normalize_local(x)

    def _local_input(
        self, s1: torch.Tensor, s2: torch.Tensor, branch: str
    ) -> torch.Tensor:
        s1_in = self._normalize_local(s1)
        s2_in = self._normalize_local(s2)
        if branch == "sar":
            s2_in = torch.zeros_like(s2_in)
        elif branch == "optical":
            s1_in = torch.zeros_like(s1_in)
        return torch.cat([s1_in, s2_in], dim=1)
