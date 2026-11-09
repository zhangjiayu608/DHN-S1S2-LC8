from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .backbone_factory import build_backbone
from .cross_modal import CrossModalResidualMappings
from .dagpid_adapter import BoundaryAwareDecoder, DAGPIDAdapter, LocalTokenEncoder
from .foundation import DINOv3TokenEncoder
from .preprocessing import CromaNormalizer

class DAGPIDSegmentor(nn.Module):
    """Frozen foundation features with complementary evidence and DAGPID adaptation."""

    def __init__(self, backbone: dict, num_classes: int=8, adapter_rank: int=64, decoder_channels: int=192, freeze_backbone: bool=True, croma_input_norm: bool=False, reliability_hidden: int=192, graph_context_dim: int=32, uncertainty_gamma: float=1.0, external_priors: dict | None=None, preprocessing: dict | None=None, head_init_seed: int=0) -> None:
        super().__init__()
        if min(num_classes, adapter_rank, decoder_channels, reliability_hidden, graph_context_dim) < 1:
            raise ValueError('Model dimensions must be positive.')
        self.backbone = build_backbone(backbone)
        self.croma_input_norm = bool(croma_input_norm)
        self.input_normalizer = CromaNormalizer(preprocessing) if self.croma_input_norm else None
        self.freeze_encoder = bool(freeze_backbone)
        if self.freeze_encoder:
            self.backbone.requires_grad_(False)
        self.external_priors = dict(external_priors or {})
        allowed_priors = {'dinov3', 'local_tokens', 'complementary', 'dagpid_fusion', 'boundary_decoder', 'local_stem'}
        if set(self.external_priors) - allowed_priors:
            raise ValueError(f'Unknown external prior settings: {sorted(set(self.external_priors) - allowed_priors)}')
        dim = int(self.backbone.encoder_dim)
        dino_cfg = dict(self.external_priors.get('dinov3', {}))
        local_cfg = dict(self.external_priors.get('local_tokens', {}))
        complement_cfg = dict(self.external_priors.get('complementary', {}))
        enabled = {'dinov3': bool(dino_cfg.pop('enabled', False)), 'local': bool(local_cfg.pop('enabled', True)), 'complementary': bool(complement_cfg.pop('enabled', True))}
        for configuration in (dino_cfg, local_cfg, complement_cfg):
            configuration.pop('enabled', None)
        self.foundation_modalities = ('sar', 'optical', 'joint')
        if enabled['dinov3']:
            self.foundation_modalities += ('dinov3',)
        if enabled['local']:
            self.foundation_modalities += ('local',)
        if enabled['complementary']:
            self.foundation_modalities += ('cf_sar', 'cf_optical')
        self.dinov3_encoder = DINOv3TokenEncoder(**dict(dino_cfg, out_dim=dim)) if enabled['dinov3'] else None
        local_cfg.pop('encoder', None)
        local_cfg.setdefault('patch_size', int(getattr(self.backbone, 'patch_size', 8)))
        self.local_token_encoder = LocalTokenEncoder(**dict(local_cfg, out_dim=dim)) if enabled['local'] else None
        self.complementary_mapping = CrossModalResidualMappings(dim=dim, **complement_cfg) if enabled['complementary'] else None
        fusion_cfg = dict(self.external_priors.get('dagpid_fusion', {}))
        self.adapter = DAGPIDAdapter(dim=dim, modalities=self.foundation_modalities, rank=adapter_rank, reliability_hidden=reliability_hidden, graph_context_dim=graph_context_dim, uncertainty_gamma=uncertainty_gamma, **fusion_cfg)
        decoder_cls = BoundaryAwareDecoder
        decoder_kwargs = dict(self.external_priors.get('boundary_decoder', {}))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(head_init_seed))
            self.decoder = decoder_cls(in_dim=dim, num_classes=num_classes, channels=decoder_channels, local_in_channels=14, use_local_stem=bool(self.external_priors.get('local_stem', True)), **decoder_kwargs)
        self.train(self.training)

    def train(self, mode: bool=True):
        super().train(mode)
        if self.freeze_encoder:
            self.backbone.eval()
        if self.dinov3_encoder is not None and self.dinov3_encoder.freeze:
            self.dinov3_encoder.model.eval()
        return self

    def set_epoch(self, epoch: int) -> None:
        self.adapter.set_epoch(epoch)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, return_aux: bool=False, source_ids: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        raw_s1, raw_s2 = (s1, s2)
        if self.input_normalizer is not None:
            s1, s2 = self.input_normalizer(s1, s2)
        with torch.set_grad_enabled(torch.is_grad_enabled() and (not self.freeze_encoder)):
            features = self.backbone(s1, s2)
        tokens = {'sar': features['SAR_encodings'], 'optical': features['optical_encodings'], 'joint': features['joint_encodings']}
        target_side = math.isqrt(next(iter(tokens.values())).shape[1])
        if target_side * target_side != next(iter(tokens.values())).shape[1]:
            raise ValueError('Foundation features must form a square token grid.')
        native_tokens = dict(tokens)
        if self.dinov3_encoder is not None:
            tokens['dinov3'] = self.dinov3_encoder(raw_s2)
        if self.local_token_encoder is not None:
            tokens['local'] = self.local_token_encoder(raw_s1, raw_s2, branch='both')
        if self.complementary_mapping is not None:
            tokens.update(self.complementary_mapping(tokens, target_side=target_side))
        fused, aux = self.adapter(tokens, target_side=target_side, source_ids=source_ids, return_aux=return_aux)
        token_map = fused.transpose(1, 2).reshape(fused.shape[0], fused.shape[2], target_side, target_side)
        decoder_kwargs = {'s1': raw_s1, 's2': raw_s2, 'branch': 'both'}
        if isinstance(self.decoder, BoundaryAwareDecoder) and 'pid_risk_token' in aux:
            decoder_kwargs['pid_risk'] = aux['pid_risk_token'].transpose(1, 2).reshape(fused.shape[0], 1, target_side, target_side)
        logits, decoder_aux = self.decoder(token_map, **decoder_kwargs)
        if logits.shape[-2:] != raw_s1.shape[-2:]:
            logits = F.interpolate(logits, size=raw_s1.shape[-2:], mode='bilinear', align_corners=False)
        output = {'logits': logits}
        if return_aux:
            aux.update(decoder_aux)
            for name, target_name in (('cf_sar', 'sar'), ('cf_optical', 'optical')):
                if name in tokens and target_name in native_tokens:
                    aux[f'{name}_tokens'] = tokens[name]
                    aux[f'{name}_target'] = native_tokens[target_name].detach()
            output['aux_both'] = aux
        return output
