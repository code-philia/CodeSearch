import numpy as np
import matplotlib.pyplot as plt

def edges_to_class_neighbors(edges, num_classes):
    """
    将边集 G.edges() 转换为 class_neighbors 的格式。
    :param edges: 邻接边集 (如 [(0, 8), (0, 2), ...])
    :param num_classes: 类别数量
    :return: class_neighbors 字典
    """
    class_neighbors = {i: [] for i in range(num_classes)}

    for edge in edges:
        class_a, class_b = edge
        if len(class_neighbors[class_a]) < 2:
            class_neighbors[class_a].append(class_b)
        if len(class_neighbors[class_b]) < 2:
            class_neighbors[class_b].append(class_a)

    return class_neighbors

def create_circular_class_layout(class_neighbors, num_classes):
    """
    创建一个基于 class_neighbors 的环形结构，并平分 [-π, π] 给每个类别。
    
    :param class_neighbors: 每个类别的邻居关系，dict结构
    :param num_classes: 总类别数
    :return: 每个类别的扇形起始角度和结束角度 (start_angle, end_angle)
    """
    visited = set()
    class_order = []  # 用于存储环形结构中类别的顺序

    def traverse_neighbors(class_id):
        """深度优先遍历，确保所有类别形成一个环形结构"""
        if class_id in visited:
            return
        visited.add(class_id)
        class_order.append(class_id)

        for neighbor in class_neighbors[class_id]:
            if neighbor not in visited:
                traverse_neighbors(neighbor)

    # 从类别0开始遍历
    traverse_neighbors(0)

    # 如果环形结构未覆盖所有类别，补全遗漏的类别
    for class_id in range(num_classes):
        if class_id not in visited:
            class_order.append(class_id)

    # 平分 [-π, π] 范围
    total_angle = 2 * np.pi  # 总角度为2π（360度）
    sector_size = total_angle / num_classes
    sector_angles = np.zeros((num_classes, 2))

    # 设置 [-π, π] 范围
    for i, class_id in enumerate(class_order):
        start_angle = -np.pi + i * sector_size
        end_angle = start_angle + sector_size
        sector_angles[class_id] = (start_angle, end_angle)

    return sector_angles


def plot_class_layout_with_tree(tree_structure, class_neighbors, weight_matrix, num_classes=10, radius=10, inner_radius=2):
    """
    从 tree_structure 中获取概率分布并进行投影，生成每个类别的扇形布局，类别扇形分布根据 class_neighbors 决定。

    :param tree_structure: 类别树结构，包含概率分布和父节点关系
    :param class_neighbors: 每个类别的邻居
    :param num_classes: 总类别数
    :param radius: 圆的最大半径（高置信度区域）
    :param inner_radius: 内圈低置信度区域的半径
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # 根据 class_neighbors 生成每个类别的扇形区域角度
    sector_angles = create_circular_class_layout(class_neighbors, num_classes)

    # 定义颜色
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    all_2d_points = []
    total_points = sum(len(tree_structure[i]) for i in range(num_classes))
    processed_points = 0

    # 遍历每个类别，并在其扇形区域内投影点
    for i in range(num_classes):
        color = colors[i]

        # 获取该类别下的所有level，按level进行批量处理
        levels = set(node['level'] for node in tree_structure[i])  # 获取所有level
        max_level = 5  # 获取最大level

        for level in levels:
            # 取出当前level的所有节点
            level_nodes = [node for node in tree_structure[i] if node['level'] == level]
            distributions = [node['distribution'] for node in level_nodes]

            # 根据当前level调整角度范围
            adjusted_start_angle, adjusted_end_angle = calculate_adjusted_angles(sector_angles, i, level, max_level)

            # 批量投影这些节点
            x_batch, y_batch = project_batch_points_in_sector(ax, sector_angles, adjusted_start_angle, adjusted_end_angle, radius, color, distributions, i, weight_matrix, level)

            # 记录批量处理的所有2D点
            all_2d_points.extend(zip(x_batch, y_batch))

            # 打印进度
            processed_points += len(level_nodes)
            print(f"Processed {processed_points} / {total_points} points")

    # 绘制低置信度的中心小圆
    inner_circle = plt.Circle((0, 0), inner_radius, color='gray', alpha=0.5, label='Low Confidence Region')
    ax.add_artist(inner_circle)

    # 绘制外部的大圆，透明，边框黑色
    outer_circle = plt.Circle((0, 0), radius, color='black', alpha=0.05, edgecolor='black', linewidth=5)  # 设置边框颜色和宽度
    ax.add_artist(outer_circle)

    ax.set_xlim(-radius * 1.2, radius * 1.2)
    ax.set_ylim(-radius * 1.2, radius * 1.2)
    plt.legend()
    plt.savefig('/home/yiming/cophi/projects/Trustvis/circle.png')
    plt.show()

    return all_2d_points


def calculate_adjusted_angles(sector_angles, class_id, level, max_level):
    """
    根据level的等级调整[adjusted_start_angle, adjusted_end_angle]的范围。
    :param sector_angles: 扇形角度范围 (start_angle, end_angle)
    :param class_id: 类别ID
    :param level: 当前level
    :param max_level: 最大的level值
    :return: 调整后的扇形角度范围 (adjusted_start_angle, adjusted_end_angle)
    """
    start_angle, end_angle = sector_angles[class_id]
    
    # 根据level与max_level的比例调整角度范围，level越高（置信度越低），占据的扇形区域越大
    if level < 6:
        level_ratio = 1 - (np.abs(level - max_level) / max_level)
    else:
        level_ratio = 1 - (np.abs((level+1) - max_level) / max_level)
    class_center_angle = (start_angle + end_angle) / 2
    new_angle_range = (end_angle - start_angle) * level_ratio
    adjusted_start_angle = class_center_angle - new_angle_range * 0.5
    adjusted_end_angle = class_center_angle + new_angle_range * 0.5

    return adjusted_start_angle, adjusted_end_angle

def project_batch_points_in_sector(ax, sector_angles, adjusted_start_angle, adjusted_end_angle, radius, color, distributions, class_id, weight_matrix, level):
    """
    批量处理一组节点，根据概率分布在扇形区域中投影多个点。

    :param ax: 画布
    :param adjusted_start_angle: 当前level的起始角度
    :param adjusted_end_angle: 当前level的结束角度
    :param radius: 半径
    :param color: 类别的颜色
    :param distributions: 当前批次的概率分布，用于决定点的位置
    """
    # 计算当前类别的中心角度
    # class_center_angle = (adjusted_start_angle + adjusted_end_angle) / 2

    class_center_angles = []

    for i in range(len(sector_angles)):
        start_angle, end_angle = sector_angles[i]
        class_center_angle = (start_angle + end_angle) / 2
        class_center_angles.append(class_center_angle)

    # 计算每个点的加权角度
    weighted_angles = []
    for distribution in distributions:
        # 使用每个类别的中心角度和对应的分布进行加权计算
        weighted_angle = sum(distribution[j] * class_center_angles[j] for j in range(len(distribution)))
        weighted_angles.append(weighted_angle)

    # 转换为NumPy数组
    weighted_angles = np.array(weighted_angles)

    # 将weighted_angles的范围映射到 [adjusted_start_angle, adjusted_end_angle] 内
    normalized_angles = np.interp(weighted_angles, 
                                   (weighted_angles.min(), weighted_angles.max()), 
                                   (adjusted_start_angle, adjusted_end_angle))


    # 计算距离到全为0.1的向量之间的距离
    distributions = np.array(distributions)
    target_vector = np.full(distributions.shape[1], 0.1)  # 创建目标向量
    distances = np.linalg.norm((distributions - target_vector) * weight_matrix[class_id], axis=1)  # 计算每个distribution到目标向量的距离
    
    # 使用当前类别的one hot vector与目标向量之间的距离进行归一化
    max_distance = np.linalg.norm((np.eye(distributions.shape[1])[class_id] - target_vector) * weight_matrix[class_id])  # 当前类别的one hot vector
    normalized_distances = distances / (max_distance + 1e-8)  # 归一化

    # 将归一化的距离映射到半径上
    r = normalized_distances * radius  # 半径根据归一化距离变化

    # 确保 x 和 y 的计算是基于正确的形状
    x = r * np.cos(normalized_angles)
    y = r * np.sin(normalized_angles)

    # 绘制点
    ax.scatter(x, y, color=color, alpha=0.3)

    return x, y
