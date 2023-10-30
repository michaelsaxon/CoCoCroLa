import torch

from cococrola.analyze.metrics.base import ImagePairMetric

class CosineSimilarity(ImagePairMetric):
    def __init__(self):
        pass

    def compute(self, image_1, image_2) -> float:
        return torch.nn.functional.cosine_similarity(image_1, image_2, dim=0).item()