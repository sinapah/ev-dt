class FederatedCoordinator:
    def __init__(self):
        self.global_weights = None

    def aggregate(self, edge_nodes):
        pass

    def ensemble_predict(self, edge_nodes, timestamp):
        return {
            site: edge_nodes[site].predict_next_arrivals(timestamp)
            for site in edge_nodes
        }
