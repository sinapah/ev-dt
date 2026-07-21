import numpy as np

from config import SITES


class FederatedCoordinator:
    def __init__(self):
        self.global_weights = None

    def aggregate(self, edge_nodes):
        pass

    def ensemble_predict(self, edge_nodes, timestamp):
        predictions = {}
        for site in SITES:
            site_preds = [edge_nodes[site].predict_next_arrivals(timestamp)]
            other_preds = [
                node.predict_next_arrivals(timestamp)
                for s, node in edge_nodes.items()
                if s != site
            ]
            all_preds = site_preds + other_preds
            predictions[site] = float(np.mean(all_preds))
        return predictions
