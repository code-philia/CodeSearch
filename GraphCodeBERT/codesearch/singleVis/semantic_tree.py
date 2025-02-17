from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def expand_tree_with_decay_one_iteration_with_adjacency(root_distribution, label_idx, max_nodes, adjacency_prob_matrix, decay_value=0.05, is_first_iteration=False):
    """
    从根节点出发，逐步生成节点直到当前标签不再是最大值，使用邻接概率矩阵进行衰减
    :param root_distribution: 当前类别的one-hot根节点概率分布
    :param label_idx: 当前类别的索引
    :param max_nodes: 生成的最大节点数
    :param adjacency_prob_matrix: 邻接概率矩阵
    :param decay_value: 衰减值
    :param is_first_iteration: 是否是首次迭代
    :return: 本次迭代生成的节点
    """
    current_node = root_distribution.copy()
    tree = []  # 初始化树

    # 在首次迭代时将根节点加入树中
    if is_first_iteration:
        tree.append(current_node)

    while len(tree) < max_nodes:
        # 对当前节点进行衰减
        new_node, continue_decay = perturb_distribution_via_decay_with_adjacency(
            current_node.copy(), label_idx, adjacency_prob_matrix, decay_value
        )

        # 如果不再继续衰减，则停止本轮迭代
        if not continue_decay:
            break

        # 将新节点加入树中，并继续衰减新的叶子节点
        tree.append(new_node)
        current_node = new_node

    return tree

# def generate_nodes_with_restart_and_adjacency(n_classes, total_nodes, adjacency_prob_matrix, decay_value=0.05):
#     """
#     针对每个类别生成节点，每次从根节点出发，直到标签不再是最大值，使用邻接概率矩阵进行衰减。
#     达到节点总数时，从根节点重新开始，而不是生成一个新的最后层级。
#     :param n_classes: 类别数量
#     :param total_nodes: 总的节点数量
#     :param adjacency_prob_matrix: 邻接概率矩阵
#     :param decay_value: 衰减值
#     :return: 包含树结构的所有类别生成的节点集合
#     """
#     nodes_per_class = total_nodes // n_classes  # 每个类别的节点数量
#     all_nodes = []  # 存储所有节点
#     tree_structure = {}  # 用于存储树结构，键为类别，值为该类别的节点树

#     for i in range(n_classes):
#         root_distribution = np.zeros(n_classes)
#         root_distribution[i] = 1.0  # 标签i的one-hot分布

#         class_nodes = []  # 存储该类别的所有节点
#         tree_structure[i] = []  # 初始化每个类别的树

#         root_node = {'distribution': root_distribution, 'parent': None, 'level': 0}  # 根节点
#         tree_structure[i].append(root_node)  # 将根节点存入类别的树中

#         class_nodes.append(root_node)  # 将根节点加入节点列表
#         current_level = 0  # 当前的层级，根节点是0级

#         # 直到生成足够的节点
#         while len(class_nodes) < nodes_per_class:
#             current_level += 1  # 进入新一层
#             previous_node = root_node  # 每一层的第一个节点的父节点应该是根节点

#             # 在这一层中生成的所有节点
#             new_nodes = expand_tree_with_decay_one_iteration_with_adjacency(
#                 root_distribution, i, nodes_per_class - len(class_nodes), adjacency_prob_matrix, decay_value
#             )

#             # 存储新生成的节点，并记录父节点信息和当前层级
#             for new_node in new_nodes:
#                 if len(class_nodes) < nodes_per_class:  # 仅当还未达标时添加新节点
#                     new_node_info = {
#                         'distribution': new_node,
#                         'parent': previous_node,  # 当前层级的父节点为上一轮的根节点或生成的节点
#                         'level': current_level  # 设置为当前的层级
#                     }
#                     tree_structure[i].append(new_node_info)  # 将新节点存入树结构
#                     class_nodes.append(new_node_info)  # 将新节点加入类别的节点列表
#                     previous_node = new_node_info  # 更新为当前节点，便于下一次迭代时作为父节点

#             # 如果已达节点数量上限，提前结束循环，避免新增额外层级
#             if len(class_nodes) >= nodes_per_class:
#                 break

#         all_nodes.extend(class_nodes)

#     return all_nodes, tree_structure  # 返回节点及其树结构



def calculate_adjacency_probabilities(ori_pred_list):
    """
    计算类与类之间的邻接概率
    :param ori_pred_list: 当前训练数据的预测语义 (n_samples, n_classes)
    :return: 邻接概率矩阵 (n_classes, n_classes)
    """
    n_classes = ori_pred_list.shape[1]
    adjacency_matrix = np.zeros((n_classes, n_classes))

    # 对每个样本计算 top1 和 top2
    top1_indices = ori_pred_list.argmax(axis=1)  # 每个样本的 top1 类别
    sorted_preds = np.argsort(-ori_pred_list, axis=1)  # 每个样本的排序，从高到低
    top2_indices = sorted_preds[:, 1]  # 每个样本的 top2 类别

    # 计算邻接概率矩阵
    for i in range(len(top1_indices)):
        top1 = top1_indices[i]
        top2 = top2_indices[i]
        adjacency_matrix[top1, top2] += 1

    # 将邻接矩阵转化为概率
    row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
    adjacency_prob_matrix = adjacency_matrix / (row_sums + 1e-8)  # 避免除以 0
    
    return adjacency_prob_matrix


def perturb_distribution_via_decay_with_adjacency(distribution, label_idx, adjacency_prob_matrix, decay_value=0.05):
    """
    扰动当前分布，从当前label的标签出发进行衰减，并根据邻接概率分配给其他标签
    :param distribution: 当前概率分布
    :param label_idx: 当前标签的索引
    :param adjacency_prob_matrix: 类别之间的邻接概率矩阵
    :param decay_value: 固定衰减值
    :return: 扰动后的新概率分布，是否继续衰减
    """
    # 如果当前标签的值不再是最大的，则停止衰减
    if distribution[label_idx] < max(distribution):
        return distribution, False

    # 选择当前label的标签进行衰减
    selected_idx = label_idx

    # 确保衰减不会导致该标签变为负值
    decay_amount = min(decay_value, distribution[selected_idx])
    distribution[selected_idx] -= decay_amount

    # 将衰减的值根据邻接概率矩阵分配给其他标签
    other_indices = [i for i in range(len(distribution)) if i != selected_idx]

    # # 获取邻接概率，根据概率选择两个不同的类别进行分配
    # adjacency_probs = adjacency_prob_matrix[selected_idx, other_indices]
    # adjacency_probs /= adjacency_probs.sum()  # 确保概率归一化

    # redistribute_idx = np.random.choice(other_indices, size=2, replace=False, p=adjacency_probs)
    redistribute_idx = np.random.choice(other_indices, size=2, replace=False)
    alpha = np.random.rand()  # 随机比例
    distribution[redistribute_idx[0]] += alpha * decay_amount
    distribution[redistribute_idx[1]] += (1 - alpha) * decay_amount

    return distribution, True


def generate_nodes_with_restart_and_adjacency(n_classes, total_nodes, adjacency_prob_matrix, decay_value=0.05):
    """
    针对每个类别生成节点，每次从根节点出发，直到标签不再是最大值，使用邻接概率矩阵进行衰减
    :param n_classes: 类别数量
    :param total_nodes: 总的节点数量
    :param adjacency_prob_matrix: 邻接概率矩阵
    :param decay_value: 衰减值
    :return: 包含树结构的所有类别生成的节点集合
    """
    nodes_per_class = total_nodes // n_classes  # 每个类别的节点数量
    tree_structure = {}  # 用于存储树结构，键为类别，值为该类别的节点树

    for i in range(n_classes):
        root_distribution = np.zeros(n_classes)
        root_distribution[i] = 1.0  # 标签i的one-hot分布

        tree_structure[i] = []  # 初始化每个类别的树
        total_generated_nodes = 0  # 已生成的节点数量
        is_first_iteration = True  # 标记是否为第一次迭代

        # 直到生成足够的节点
        while total_generated_nodes < nodes_per_class:
            current_node = root_distribution.copy()  # 每次从根节点开始
            parent_node_info = {'distribution': current_node, 'parent': None, 'level': 0}

            # 只有在第一次迭代时，记录根节点
            if is_first_iteration:
                tree_structure[i].append(parent_node_info)
                is_first_iteration = False

            # 开始从根节点生成路径，直到无法继续衰减
            while total_generated_nodes < nodes_per_class:
                # 调用衰减函数生成新节点
                new_node, continue_decay = perturb_distribution_via_decay_with_adjacency(
                    current_node.copy(), i, adjacency_prob_matrix, decay_value
                )

                # 如果衰减停止，不再生成子节点，跳出循环
                if not continue_decay:
                    break

                # 生成新的子节点并更新树结构
                new_node_info = {
                    'distribution': new_node,
                    'parent': parent_node_info,  # 当前生成节点的父节点
                    'level': parent_node_info['level'] + 1  # 设置为父节点的level + 1
                }
                tree_structure[i].append(new_node_info)

                # 更新当前节点信息并递增生成的节点数量
                parent_node_info = new_node_info
                current_node = new_node
                total_generated_nodes += 1

                # 如果节点数量已经达到目标，跳出循环
                if total_generated_nodes >= nodes_per_class:
                    break

    return tree_structure  # 返回树结构


def generate_multi_branch_tree_with_restart(n_classes, total_nodes, branching_factor, adjacency_prob_matrix, decay_value=0.05):
    """
    针对每个类别生成多叉树节点，每个父节点生成多个子节点，直到标签不再是最大值，使用邻接概率矩阵进行衰减。
    若所有当前层级的节点都无法继续衰减，则重新从根节点生成新节点，直到生成总的目标节点数。
    
    :param n_classes: 类别数量
    :param total_nodes: 总的节点数量
    :param branching_factor: 每个父节点生成的子节点数量
    :param adjacency_prob_matrix: 邻接概率矩阵
    :param decay_value: 衰减值
    :return: 包含树结构的所有类别生成的节点集合
    """
    nodes_per_class = total_nodes // n_classes  # 每个类别的节点数量
    tree_structure = {}  # 用于存储树结构，键为类别，值为该类别的节点树

    for i in range(n_classes):
        root_distribution = np.zeros(n_classes)
        root_distribution[i] = 1.0  # 标签i的one-hot分布

        tree_structure[i] = []  # 初始化每个类别的树
        total_generated_nodes = 0  # 已生成的节点数量
        is_first_iteration = True  # 标记是否为第一次迭代

        # 当前层级的节点列表
        current_level_nodes = []
        root_node_info = None  # 保存根节点信息

        while total_generated_nodes < nodes_per_class:
            next_level_nodes = []  # 存储下一层级的节点

            # 第一次迭代时生成根节点
            if is_first_iteration:
                root_node_info = {'distribution': root_distribution.copy(), 'parent': None, 'level': 0}
                tree_structure[i].append(root_node_info)
                current_level_nodes.append(root_node_info)
                total_generated_nodes += 1
                is_first_iteration = False

            # 遍历当前层级的节点，为它们生成子节点
            for parent_node_info in current_level_nodes:
                # 为当前父节点生成多个子节点
                for _ in range(branching_factor):
                    if total_generated_nodes >= nodes_per_class:
                        break

                    # 调用衰减函数生成新节点
                    new_node, continue_decay = perturb_distribution_via_decay_with_adjacency(
                        parent_node_info['distribution'].copy(), i, adjacency_prob_matrix, decay_value
                    )

                    # 如果无法继续衰减，跳过该父节点
                    if not continue_decay:
                        continue

                    # 生成新的子节点并更新树结构
                    new_node_info = {
                        'distribution': new_node,
                        'parent': parent_node_info,  # 当前生成节点的父节点
                        'level': parent_node_info['level'] + 1  # 设置为父节点的level + 1
                    }
                    tree_structure[i].append(new_node_info)
                    next_level_nodes.append(new_node_info)
                    total_generated_nodes += 1

                    # 如果达到节点数量上限，跳出循环
                    if total_generated_nodes >= nodes_per_class:
                        break

            # 如果这一层所有节点都无法继续衰减，则重新从根节点生成新节点
            if not next_level_nodes:
                # 再次从根节点生成新节点
                new_node, _ = perturb_distribution_via_decay_with_adjacency(root_distribution.copy(), i, adjacency_prob_matrix, decay_value)
                new_node_info = {
                    'distribution': new_node,
                    'parent': root_node_info,  # 根节点的子节点应当指向根节点作为父节点
                    'level': 1  # 新生成的节点处于第1层
                }
                tree_structure[i].append(new_node_info)
                next_level_nodes.append(new_node_info)
                total_generated_nodes += 1

            # 更新当前层级节点为下一层级节点
            current_level_nodes = next_level_nodes

    return tree_structure  # 返回树结构


def split_node(distribution, label_idx, step=0.1):
    """
    对当前分布进行分裂，确保主类别保持最大值。
    
    :param distribution: 当前类别分布
    :param label_idx: 当前标签的索引
    :param step: 每次分裂递减的阈值
    :return: 新生成的符合条件的子节点列表
    """
    children = []
    current_value = distribution[label_idx]

    if current_value <= step:
        return children  # 如果当前主类别值无法再递减，停止生成

    # 对于每一个其他类别，生成一个新的子节点
    for i in range(len(distribution)):
        if i != label_idx:
            new_distribution = distribution.copy()
            new_distribution[label_idx] -= step
            new_distribution[i] += step
            
            # 在这里判断新生成的节点是否满足主类别仍然最大
            if new_distribution[label_idx] >= max(new_distribution):
                children.append(new_distribution)

    return children


def generate_tree(n_classes, max_depth, step=0.1):
    """
    根据递减策略生成每个类别的树结构。
    
    :param n_classes: 类别数量
    :param max_depth: 生成树的最大深度
    :param step: 每次分裂的阈值
    :return: 每个类别生成的树结构
    """
    tree_structure = {i: [] for i in range(n_classes)}  # 用于存储每个类别的树

    # 初始化根节点（one-hot向量）
    root_nodes = [np.eye(n_classes)[i] for i in range(n_classes)]

    # 遍历每一个类别的根节点
    for label_idx in range(n_classes):
        print(f"Generating tree for class {label_idx}")
        root_node = root_nodes[label_idx]
        # 添加根节点
        tree_structure[label_idx].append({'distribution': root_node, 'parent': [], 'level': 0})
        current_level_nodes = [{'distribution': root_node, 'parent': [], 'level': 0}]

        for level in range(1, max_depth + 1):
            next_level_nodes = []
            
            # 遍历当前层级的所有节点
            for parent_node in current_level_nodes:
                parent_distribution = parent_node['distribution']

                # 判断当前主类别是否保持最大值
                if parent_distribution[label_idx] >= max(parent_distribution):
                    # 对当前节点进行分裂
                    children = split_node(parent_distribution, label_idx, step)

                    for child_distribution in children:
                        # 查看该分布是否已存在，避免重复生成
                        for existing_node in tree_structure[label_idx]:
                            if np.allclose(existing_node['distribution'], child_distribution):
                                # 已存在，合并父节点（添加到现有的父节点列表中）
                                if not any(np.array_equal(parent_node['distribution'], p) for p in existing_node['parent']):
                                    existing_node['parent'].append(parent_node['distribution'])
                                break
                        else:
                            # 新节点，添加到树结构，初始化父节点为当前节点
                            child_node = {
                                'distribution': child_distribution, 
                                'parent': [parent_node['distribution']], 
                                'level': level
                            }
                            tree_structure[label_idx].append(child_node)
                            next_level_nodes.append(child_node)

            if not next_level_nodes:
                break  # 如果没有生成新节点，停止生成
            current_level_nodes = next_level_nodes
            print(f"Level {level} has {len(current_level_nodes)} nodes")

    return tree_structure


def generate_and_merge_tree_by_level(n_classes, max_depth, step=0.1):
    """
    分层生成每个类别的树结构，每生成一层后进行合并操作，并更新当前层节点。
    
    :param n_classes: 类别数量
    :param max_depth: 树的最大深度
    :param step: 每次分裂的阈值
    :return: 每个类别生成的树结构
    """
    tree_structure = {i: [] for i in range(n_classes)}  # 用于存储每个类别的树
    parent_map = {}  # 用于存储父子节点关系

    # 初始化根节点（one-hot向量）
    root_nodes = [np.eye(n_classes)[i] for i in range(n_classes)]

    # 遍历每一个类别的根节点
    for label_idx in range(n_classes):
        print(f"Generating tree for class {label_idx}")
        root_node = root_nodes[label_idx]
        tree_structure[label_idx].append({'distribution': root_node, 'parent': None, 'level': 0})
        current_level_nodes = [{'distribution': root_node, 'parent': None, 'level': 0}]
        
        for level in range(1, max_depth + 1):
            next_level_nodes = []

            # 遍历当前层级的所有节点
            for parent_node in current_level_nodes:
                parent_distribution = parent_node['distribution']

                # 判断当前主类别是否保持最大值
                if parent_distribution[label_idx] >= max(parent_distribution):
                    # 对当前节点进行分裂
                    children = split_node(parent_distribution, label_idx, step)

                    for child_distribution in children:
                        child_node = {'distribution': child_distribution, 'parent': [parent_node['distribution']], 'level': level}
                        next_level_nodes.append(child_node)
                        parent_map[tuple(child_distribution)] = [parent_node['distribution']]

            # 对当前层级的节点进行合并
            merged_next_level_nodes = merge_and_update_nodes_in_level(next_level_nodes)
            tree_structure[label_idx].extend(merged_next_level_nodes)

            if not merged_next_level_nodes:
                break  # 如果没有生成新节点，停止生成

            current_level_nodes = merged_next_level_nodes
            # print(f"Level {level} generated {len(current_level_nodes)} nodes, total: {len(tree_structure[label_idx])}")

    return tree_structure, parent_map


def merge_and_update_nodes_in_level(next_level_nodes):
    """
    对下一层的节点进行合并并更新，确保其基于合并后的节点。
    
    :param next_level_nodes: 下一层的节点集合
    :return: 更新后的下一层节点
    """
    merged_nodes = []
    node_map = {}

    for node in next_level_nodes:
        key = tuple(node['distribution'])
        if key in node_map:
            # 如果节点已存在，合并父节点信息
            existing_node = node_map[key]
            existing_node['parent'].extend(node['parent'])
        else:
            # 如果是新节点，则添加到合并节点列表
            node_map[key] = node
            merged_nodes.append(node)

    return merged_nodes


def merge_across_classes_and_remove_duplicates(tree_structure, decimal_places=5):
    """
    在类别之间合并重复的节点，保留最早出现的节点并删除其他类别中的重复节点。
    
    :param tree_structure: 每个类别生成的树结构
    :param decimal_places: 舍入到小数点后多少位
    :return: 更新后的 tree_structure
    """
    seen_distributions = set()  # 使用集合来跟踪出现过的分布
    merged_tree_structure = {i: [] for i in tree_structure.keys()}  # 合并后的树结构

    # 遍历所有类别的树
    for class_id, nodes in tree_structure.items():
        for node in nodes:
            # 对分布进行四舍五入并转换为 tuple 以用于集合判断
            rounded_distribution = tuple(np.round(node['distribution'], decimal_places))

            if rounded_distribution not in seen_distributions:
                # 如果该节点第一次出现，保留它，并记录它的舍入分布
                seen_distributions.add(rounded_distribution)
                merged_tree_structure[class_id].append(node)
            else:
                # 如果该节点已经存在，跳过它
                continue

    return merged_tree_structure





