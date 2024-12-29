import numpy as np
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity


def CosDefense(old_weight, new_weights_dict):
    new_weights = []
    for w_local in new_weights_dict:
        params = [val.cpu().numpy() for _, val in w_local.items()]
        new_weights.append(params)

    global_last_layer = old_weight[-2].reshape(-1)
    last_layer_grads = []

    # Calculate gradients
    for new_weight in new_weights:
        last_layer_grads.append(new_weight[-2].reshape(-1) - global_last_layer)

    # Check for NaN values in gradients
    if np.any(np.isnan(last_layer_grads)):
        print("NaN detected in last_layer_grads!")
        last_layer_grads = np.nan_to_num(last_layer_grads)  # Replace NaNs with 0

    # Check for NaN values in the global last layer
    if np.any(np.isnan(global_last_layer)):
        print("NaN detected in global_last_layer!")
        global_last_layer = np.nan_to_num(global_last_layer)  # Replace NaNs with 0

    # Compute cosine similarity
    scores = np.abs(cosine_similarity(np.array(last_layer_grads), [global_last_layer]).reshape(-1))

    # Normalize scores
    min_score = np.min(scores)
    scores = (scores - min_score) / (np.max(scores) - min_score)

    # Set threshold
    threshold = np.mean(scores)  # or np.mean(scores) + 0.5 * np.std(scores)

    # Identify benign clients
    benign_indices = scores < threshold

    # Weighting and aggregation
    weight = 1 / (sum(benign_indices))
    fractions = benign_indices * weight
    weighted_weights = [
        [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
    ]

    # Compute average weights of each layer
    aggregate_weights = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]

    return aggregate_weights, scores, benign_indices