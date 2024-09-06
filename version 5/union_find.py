import numpy as np

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            # Union by rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def find_clusters(distance_matrix, ids, threshold):
    distance_matrix = np.array(distance_matrix)
    num_items = distance_matrix.shape[0]
    
    uf = UnionFind(num_items)
    
    # Union items based on distance
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if distance_matrix[i, j] < threshold:
                uf.union(i, j)
    clusters = {}
    for i in range(num_items):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(ids[i])  # Map index to ids
    
    return list(clusters.values())

# # Example usage:
# distance_matrix = [
#     [0, 2, 10, 7],
#     [2, 0, 6, 4],
#     [10, 6, 0, 3],
#     [7, 4, 3, 0]
# ]

