import numpy as np
import networkx as nx

def create_optimized_adjacency_graph(adjacency_prob_matrix, n_neighbors=2):
    """
    根据邻接概率矩阵创建邻接图，确保每个类有且仅有两个邻居，并通过淘汰次优邻接寻找最优解。
    
    :param adjacency_prob_matrix: 邻接概率矩阵 (n_classes, n_classes)
    :param n_neighbors: 每个类邻接的类数目
    :return: 图结构 G
    """
    n_classes = adjacency_prob_matrix.shape[0]
    G = nx.Graph()

    # 跟踪每个类已经有多少邻居
    neighbors_count = {i: 0 for i in range(n_classes)}

    # 用于记录每个类优先连接的候选邻居，按概率从大到小排序
    candidates = {i: np.argsort(-adjacency_prob_matrix[i]) for i in range(n_classes)}
    
    def find_next_best_neighbor(i, excluded_neighbors):
        """
        在候选邻居中，找到当前未连接的最优邻居
        """
        for neighbor in candidates[i]:
            if neighbor not in excluded_neighbors and neighbors_count[neighbor] < n_neighbors:
                return neighbor
        return None  # 如果没有可用邻居
    
    # 连接每个类的两个最优邻居
    for i in range(n_classes):
        connected_neighbors = set()
        
        # 遍历候选邻居，确保不超过n_neighbors个连接
        for neighbor in candidates[i]:
            if neighbors_count[i] < n_neighbors and neighbors_count[neighbor] < n_neighbors:
                G.add_edge(i, neighbor)
                neighbors_count[i] += 1
                neighbors_count[neighbor] += 1
                connected_neighbors.add(neighbor)
            
            # 如果邻居数已满，跳出循环
            if neighbors_count[i] >= n_neighbors:
                break
        
        # 检查当前类的邻居数量是否超过了限制，如果超出，则需要进行回溯调整
        if neighbors_count[i] > n_neighbors:
            # 找到概率最小的邻居进行淘汰
            sorted_neighbors = sorted(connected_neighbors, key=lambda x: adjacency_prob_matrix[i, x])
            while neighbors_count[i] > n_neighbors:
                # 淘汰最小概率的邻居
                to_remove = sorted_neighbors.pop(0)
                G.remove_edge(i, to_remove)
                neighbors_count[i] -= 1
                neighbors_count[to_remove] -= 1
                
                # 为被淘汰的邻居寻找新的邻居
                new_neighbor = find_next_best_neighbor(to_remove, connected_neighbors)
                if new_neighbor is not None:
                    G.add_edge(to_remove, new_neighbor)
                    neighbors_count[to_remove] += 1
                    neighbors_count[new_neighbor] += 1
                    connected_neighbors.add(new_neighbor)

    return G