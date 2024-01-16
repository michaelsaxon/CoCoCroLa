import os
from typing import List, Union
from PIL import Image, ImageFile

import numpy as np
import torch


class ImagePairMetric():
    """Base class for image pair metrics."""

    def __init__(self):
        raise NotImplementedError

    # Given two images, return the metric, images should be vectorized representations, eg
    # - torch.Tensor
    # - numpy.ndarray
    def compute(self, image_1, image_2) -> float:
        raise NotImplementedError

    # Given a pair of same length lists of images, compute the pairwise score over every pair
    def compute_from_pair(self, image_list_1, image_list_2) -> float:
        scores = np.zeros(len(image_list_1), len(image_list_2))
        for i in range(len(image_list_1)):
            for j in range(len(image_list_2)):
                scores[i][j] = self.compute(image_list_1[i], image_list_2[j])
        return scores
        
        
# results_dict contains a list of vectors for each language