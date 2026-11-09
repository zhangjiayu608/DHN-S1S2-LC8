from __future__ import annotations

from torch import nn

from .anysat_backbone import build_anysat_backbone
from .croma_backbone import build_backbone as build_croma_backbone
from .copernicusfm_backbone import build_copernicusfm_backbone
from .fomo_backbone import build_fomo_backbone
from .galileo_backbone import build_galileo_backbone
from .mars_backbone import build_mars_backbone
from .panopticon_backbone import build_panopticon_backbone
from .ramen_backbone import build_ramen_backbone
from .skysense_backbone import build_skysense_backbone
from .softcon_backbone import build_softcon_backbone
from .tessera_backbone import build_tessera_backbone
from .terramind_backbone import build_terramind_backbone


def build_backbone(config: dict) -> nn.Module:
    name = str(config.get("name", "croma")).lower()
    if name in {"anysat", "any-sat"}:
        return build_anysat_backbone(config)
    if name in {"croma"}:
        return build_croma_backbone(config)
    if name in {"copernicusfm", "copernicus-fm", "copernicus_fm"}:
        return build_copernicusfm_backbone(config)
    if name in {"fomo", "fomo-net", "fomonet"}:
        return build_fomo_backbone(config)
    if name in {"galileo"}:
        return build_galileo_backbone(config)
    if name in {"mars", "ma-rs"}:
        return build_mars_backbone(config)
    if name in {"panopticon", "panopticon-vitb14", "panopticon_vitb14"}:
        return build_panopticon_backbone(config)
    if name in {"ramen"}:
        return build_ramen_backbone(config)
    if name in {"skysense", "skysense++", "skysenseplusplus", "skysense-plusplus"}:
        return build_skysense_backbone(config)
    if name in {"softcon", "soft-con"}:
        return build_softcon_backbone(config)
    if name in {"tessera"}:
        return build_tessera_backbone(config)
    if name in {"terramind", "terra-mind", "terra_mind"}:
        return build_terramind_backbone(config)
    raise ValueError(f"Unknown backbone: {name}")
