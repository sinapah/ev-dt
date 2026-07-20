import copy
import numpy as np


class FederatedCoordinator:
    def __init__(self):
        self.global_weights = None

    def aggregate(self, edge_nodes):
        all_weights = [node.get_weights() for node in edge_nodes]
        trained = [w for w in all_weights if w is not None]
        if not trained:
            return
        if len(trained) < len(edge_nodes):
            all_weights = trained
        n_clients = len(all_weights)
        avg = {}
        for key in all_weights[0]:
            avg[key] = np.mean(
                [w[key] for w in all_weights],
                axis=0
            )
        self.global_weights = avg
        for node in edge_nodes:
            node.set_weights(copy.deepcopy(avg))
