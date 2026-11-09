from .segmentor import DAGPIDSegmentor


def build_segmentor(config: dict):
    return DAGPIDSegmentor(**config)


__all__ = ["build_segmentor", "DAGPIDSegmentor"]
