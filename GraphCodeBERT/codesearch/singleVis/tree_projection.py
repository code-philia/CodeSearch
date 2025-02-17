import umap
from umap.umap_ import fuzzy_simplicial_set
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix
from pynndescent import NNDescent

def project_tree_structure_level_by_level(tree_structure, n_classes, adjacency_prob_matrix_array, n_neighbors=2, min_dist=0.1, metric='euclidean'):
    """
    按照层级逐步投影树结构，保持前层节点不变，将新节点投影到已有布局中，并且自定义邻居图。
    
    :param tree_structure: 树结构，每个类别的节点包含 'distribution', 'parent', 'level'
    :param n_classes: 类别数量
    :return: 所有节点的二维投影 (包含层次信息)
    """
    initial_lr = 0.01
    lr_decay=0.1
    # 初始化UMAP模型
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=1, metric=metric, learning_rate=initial_lr)

    all_high_dim_points = []  # 存储所有高维点
    all_2d_points = []  # 存储对应的二维投影点
    all_classes = []  # 存储所有点的类别
    # neighbor_graph = []  # 用于构建自定义邻居图

    all_high_dim_history = []  # 存储每一轮的 all_high_dim_points
    all_2d_history = []  # 存储每一轮的 all_2d_points

    knn_indices = []  # 存储最近邻索引
    knn_dists = []  # 存储最近邻距离

    # 记录每个level的二维点
    all_levels_2d_projection = {}

    # 找出树中所有存在的最大层数
    max_level = max(node['level'] for nodes in tree_structure.values() for node in nodes)
    current_lr = initial_lr
    # 遍历每一层，从0开始投影
    for level in range(max_level + 1):

        current_high_dim_points = []  # 当前层的高维点
        current_classes = []  # 记录当前层每个点的类别
        print(f"Projecting Level {level} with learning rate {current_lr}")

        # 遍历每个类别
        for class_id in range(n_classes):
            class_tree = tree_structure[class_id]
            
            # 获取该类别中当前level的节点
            current_level_nodes = [node for node in class_tree if node['level'] == level]

            # 记录每个点的类别
            current_classes.extend([class_id] * len(current_level_nodes))

            # 对于该类中的每个点，建立它们的邻居关系
            class_level_indices = list(range(len(all_high_dim_points) + len(current_high_dim_points), len(all_high_dim_points) + len(current_high_dim_points) + len(current_level_nodes)))
            # print(class_level_indices)
            # 强制连接点和它们的父节点
                    
            for idx, node in zip(class_level_indices, current_level_nodes):
                neighbors = []  # 存储该节点的邻居索引
                distances = []  # 存储该节点的邻居距离

                if node['parent'] is not None:
                    # 找到父节点在 all_high_dim_points 中的位置
                    parent_distribution = node['parent']['distribution']
                    parent_index = next((i for i, p in enumerate(all_high_dim_points) if np.array_equal(p, parent_distribution)), None)
                    
                    if parent_index is not None:
                        # 连接当前节点和父节点
                        neighbors.append(parent_index)
                        # neighbor_graph.append([idx, parent_index])
                        # neighbor_graph.append([parent_index, idx])

                        # 计算当前节点和父节点的距离
                        parent_distance = np.linalg.norm(np.array(node['distribution']) - np.array(node['parent']['distribution']))
                        distances.append(parent_distance)

                        # # 在同一层中找到最近的其他节点
                        # min_distance = float('inf')
                        # nearest_neighbor = None
                        # weight_vector = adjacency_prob_matrix_array[class_id]

                        # for other_idx, other_node in zip(class_level_indices, current_level_nodes):
                        #     if other_idx != idx:  # 确保不与自己比较
                        #         # 计算加权后的距离，先计算两个节点分布的差值
                        #         diff = np.array(node['distribution']) - np.array(other_node['distribution'])
                                
                        #         # 对差值的每个维度乘以对应的权重
                        #         weighted_diff = diff * weight_vector
                                
                        #         # 计算加权后的距离
                        #         dist = np.linalg.norm(weighted_diff)

                        #         if dist < min_distance:
                        #             min_distance = dist
                        #             nearest_neighbor = other_idx

                        # # 连接到最近邻节点
                        # if nearest_neighbor is not None:
                        #     neighbors.append(nearest_neighbor)
                        #     distances.append(min_distance)    

                        # 环形结构
                        origin_len = len(all_high_dim_points) + len(current_high_dim_points)
                        local_idx = idx-origin_len
                        next_idx = (local_idx + 1) % len(class_level_indices)  # 下一个节点，最后一个连接到第一个
                        # 连接当前节点和下一个节点
                        neighbors.append(next_idx+origin_len)
                        
                        # 计算当前节点和下一个节点的距离
                        next_distance = np.linalg.norm(np.array(current_level_nodes[local_idx]['distribution']) - np.array(current_level_nodes[(next_idx)]['distribution']))
                        distances.append(next_distance)

                        # # 连接当前节点和它的兄弟节点（同父节点的其他子节点）
                        # sibling_nodes = [sibling_node for sibling_node in current_level_nodes 
                        #                 if sibling_node['parent'] is not None and 
                        #                 np.array_equal(sibling_node['parent']['distribution'], node['parent']['distribution'])]

                        # sibling_indices = []
                        # for sibling_node in sibling_nodes:
                        #     for sibling_idx, sibling in enumerate(current_level_nodes):
                        #         if np.array_equal(sibling['distribution'], sibling_node['distribution']):
                        #             sibling_indices.append(class_level_indices[sibling_idx])

                        # # # 排除自己，连接到所有兄弟节点
                        # # for sibling_idx in sibling_indices:
                        # #     if sibling_idx != idx:
                        # #         neighbors.append(sibling_idx)
                        # #         # neighbor_graph.append([idx, sibling_idx])
                        # #         # neighbor_graph.append([sibling_idx, idx])

                        # #         # 计算当前节点和兄弟节点的距离
                        # #         local_sibling_idx = sibling_idx - (len(all_high_dim_points) + len(current_high_dim_points))
                        # #         sibling_distance = np.linalg.norm(np.array(node['distribution']) - np.array(current_level_nodes[local_sibling_idx]['distribution']))
                        # #         distances.append(sibling_distance)

                        # # 首尾相连的逻辑：将兄弟节点排序，找到和当前idx最接近的兄弟节点
                        # if len(sibling_indices) > 1:
                        #     sibling_indices_sorted = sorted(sibling_indices)  # 按索引排序

                        #     # 找到当前 idx 的位置
                        #     idx_position = sibling_indices_sorted.index(idx)

                        #     # 找到前一个和后一个兄弟节点的索引
                        #     # prev_sibling_idx = sibling_indices_sorted[idx_position - 1]  # 前一个节点
                        #     next_sibling_idx = sibling_indices_sorted[(idx_position + 1) % len(sibling_indices_sorted)]  # 后一个节点（循环连接）

                        #     # 只连接到前一个和后一个兄弟节点，形成环形结构
                        #     # for sibling_idx in [prev_sibling_idx, next_sibling_idx]:
                        #     if next_sibling_idx != idx:
                        #         neighbors.append(next_sibling_idx)
                        #         # neighbor_graph.append([idx, sibling_idx])
                        #         # neighbor_graph.append([sibling_idx, idx])

                        #         # 计算当前节点和兄弟节点的距离
                        #         local_sibling_idx = next_sibling_idx - (len(all_high_dim_points) + len(current_high_dim_points))
                        #         sibling_distance = np.linalg.norm(np.array(node['distribution']) - np.array(current_level_nodes[local_sibling_idx]['distribution']))
                        #         distances.append(sibling_distance)

                while len(neighbors) < n_neighbors:
                    neighbors.append(idx)  # 将自己添加为邻居
                    distances.append(0.0) 
                # 将邻居索引和距离保存到 knn_indices 和 knn_dists 中
                if len(neighbors) > 0:
                    # print(distances)
                    knn_indices.append(neighbors)  # 确保只保留 n_neighbors 个邻居
                    knn_dists.append(distances)   # 第一个邻居是自己，距离为 0

            # 将当前层的高维点加入
            current_high_dim_points.extend([node['distribution'] for node in current_level_nodes])
            # print(class_id)
            # print(neighbor_graph[-10:])
        # 转换为numpy数组
        current_high_dim_points = np.array(current_high_dim_points)
        current_classes = np.array(current_classes)
        knn_indices_array = np.array([np.array(x) if isinstance(x, list) else x for x in knn_indices])
        knn_dists_array = np.array([np.array(x) if isinstance(x, list) else x for x in knn_dists])
        # 第一次，直接投影根节点（Level 0）
        if level == 0:
            initial_2d_projection = umap_model.fit_transform(current_high_dim_points)
            all_levels_2d_projection[level] = initial_2d_projection
            all_high_dim_points.extend(current_high_dim_points)
            all_2d_points.extend(initial_2d_projection)
            all_classes.extend(current_classes)
        else:
            # if level > 6:
            #     break
            # 后续层次，使用前面的二维投影作为init
            previous_2d_projection = np.vstack(all_2d_points)  # 获取当前已有点的二维投影

            # # 为新层提供已有投影作为初始值
            # init_layout = np.vstack((previous_2d_projection, np.random.randn(len(current_high_dim_points), 2)))  # 用随机二维点作为新层初始值

            # 拼接高维点
            combined_high_dim_points = np.vstack((all_high_dim_points, current_high_dim_points))

            # 计算 fuzzy simplicial set，手动将自定义的邻居图传递给 UMAP
            umap_model.min_dist = min_dist
            umap_model.learning_rate = current_lr

            n_neighbors = len(knn_indices[0])
            # print(knn_dists_array)
            # print(len(combined_high_dim_points), len(knn_indices))
            rows = np.repeat(np.arange(len(combined_high_dim_points)), n_neighbors)  # 每个样本的行索引，重复 n_neighbors 次
            cols = knn_indices_array.flatten()  # 每个样本最近邻的列索引
            values = knn_dists_array.flatten()
            # print(cols)
            sparse_distance_matrix = csr_matrix((values, (rows, cols)), shape=(len(combined_high_dim_points), len(combined_high_dim_points)))
            # print(knn_indices_array)
            # print(knn_dists_array)
            # print(sparse_distance_matrix)
            nnd = NNDescent(combined_high_dim_points, n_neighbors=n_neighbors, metric="euclidean")
            umap_model.metric='precomputed'
            umap_model.precomputed_knn=(knn_indices_array, knn_dists_array, nnd)
            umap_model.repulsion_strength = 3
            umap_model.negative_sample_rate = 20

            # umap_model.graph_, sigmas, rhos = fuzzy_simplicial_set(
            #     X=combined_high_dim_points,
            #     n_neighbors=n_neighbors,
            #     metric=metric,
            #     random_state=42,
            #     knn_indices=knn_indices_array,
            #     knn_dists=knn_dists_array,
            # )

            # 运行 fit_transform
            symmetric_matrix = (sparse_distance_matrix + sparse_distance_matrix.T) / 2
            current_2d_projection = umap_model.fit_transform(symmetric_matrix)

            # 更新所有高维和二维点
            all_high_dim_points = combined_high_dim_points
            all_2d_points = current_2d_projection
            all_classes = np.hstack((all_classes, current_classes))
            all_levels_2d_projection[level] = current_2d_projection[len(previous_2d_projection):]  # 只记录当前层的点

        current_lr *= lr_decay
        # 记录每一轮的高维点和二维点
        all_high_dim_history.append(np.array(all_high_dim_points))
        all_2d_history.append(np.array(all_2d_points))

    return all_high_dim_points, all_2d_points, all_high_dim_history, all_2d_history
