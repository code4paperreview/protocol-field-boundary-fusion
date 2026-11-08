"""Log-domain evaluation of the author's unchanged PFT product objective."""

import math

import numpy as np
import pandas as pd

from .pft import FEATURES, tree_nodes


def label_tree(root, model, lengths, author):
    nodes = [node for node in tree_nodes(root) if node.id != "root"]
    classes = [name for name, _ in sorted(lengths.items())]
    if list(model.classes_) != classes:
        raise ValueError("Classifier columns must match the author's sorted syntax classes")
    root.type, root.features, root.probas, root.best_prob = None, None, None, None
    matrix = []
    for node in nodes:
        total = sum(node.trans)
        transitions = [value / total if total > 0 else value for value in node.trans]
        matrix.append(transitions + author.avg_bit_from_pairs(node.time_byte_pairs))
    features = pd.DataFrame(matrix, columns=FEATURES)
    for start in range(0, len(nodes), 4096):
        probabilities = model.predict_proba(features.iloc[start:start + 4096])
        for node, probability in zip(nodes[start:start + 4096], probabilities):
            node.probas = probability.reshape(1, -1)

    minimum_depth = {}
    for node in reversed(nodes):
        minimum_depth[node] = 1 + min((minimum_depth[c] for c in node.children), default=0)
        node.type, node.best_log_probability = "UINT8", -math.inf
        for index, (syntax, width) in enumerate(sorted(lengths.items())):
            if minimum_depth[node] < width:
                continue
            paths = author.get_paths_with_n_nodes(node, width)
            if len(paths) == 1 and paths[0][0].nc != paths[0][-1].nc:
                continue
            score = 0.0
            for path in paths:
                for part in path:
                    probability = float(part.probas[0, index])
                    score += math.log(probability) if probability > 0 else -math.inf
                score += math.log(1 / 3)  # Same author penalty, once per path.
                for continuation in path[-1].children:
                    score += continuation.best_log_probability
            if score > node.best_log_probability:
                node.type, node.best_log_probability = syntax, score
        node.best_prob = math.exp(node.best_log_probability)
    return {
        "probability_computation": "author product objective in log domain; same syntax widths, paths, guards and 1/3 penalty",
        "classifier_prediction": "identical 16 features; batches of at most 4096 nodes",
        "zero_best_probability_nodes": sum(node.best_prob == 0 for node in nodes),
        "finite_log_probability_nodes": sum(math.isfinite(node.best_log_probability) for node in nodes),
        "minimum_class_probability": float(min(np.min(node.probas) for node in nodes)),
    }
