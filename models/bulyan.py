from itertools import combinations

import tools
import math
import torch
import numpy as np


class Bulyan:
    def __init__(self, f=10, m=None):
        self.f = f  # Number of Byzantine workers to tolerate
        self.m = m  # Optional, used in multi-krum

    def aggregate(self, parameters_dicts):
        if self.m is None:
            self.m = len(parameters_dicts) - 2 * self.f - 2

        # Get the list of keys from the first OrderedDict
        keys = list(parameters_dicts[0].keys())

        # Initialize an OrderedDict to store aggregated parameters
        aggregated_params = torch.nn.ParameterDict()

        # Process each key (i.e., each layer)
        for key in keys:
            # Extract the layer-specific parameters from each OrderedDict
            layer_params = [params[key] for params in parameters_dicts]
            aggregated_params[key.replace('.', '_')] = self.aggregate_layer(layer_params)  # Replace dots in key names

        return aggregated_params

    def aggregate_layer(self, layer_params):
        n = len(layer_params)
        flattened_params = [p.flatten() for p in layer_params]
        stacked_params = torch.stack(flattened_params)

        # Compute pairwise distances
        distances = torch.cdist(stacked_params, stacked_params, p=2)

        # Apply Multi-Krum algorithm to select m best models
        scores = torch.zeros(n)
        for i in range(n):
            top_m_scores = torch.topk(distances[i], self.m + 2 * self.f, largest=False).values
            scores[i] = top_m_scores.sum()

        selected_indices = torch.topk(scores, self.m, largest=False).indices
        selected_params = stacked_params[selected_indices]

        # Apply Bulyan algorithm: median of selected parameters
        median = torch.median(selected_params, dim=0).values

        # Reshape to original shape if needed
        original_shape = layer_params[0].shape
        aggregated = median.view(original_shape)

        return aggregated

