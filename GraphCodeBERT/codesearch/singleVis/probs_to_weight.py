import numpy as np

def create_weight_matrix(adjacency_prob_matrix):
    """
    对非对角线元素进行倒数操作，并将结果 normalize 为权重矩阵，对角线置 1，
    每一行的和为 2。
    
    :param adjacency_prob_matrix: 原始邻接概率矩阵 (n_classes, n_classes)
    :return: 处理后的近距离权重矩阵
    """
    n_classes = adjacency_prob_matrix.shape[0]
    
    # 初始化倒数矩阵
    inv_matrix = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        # 对非对角线元素取倒数
        non_diag = np.copy(adjacency_prob_matrix[i, :])
        non_diag[i] = 0  # 确保对角线元素不参与处理
        inv_matrix[i, :] = np.where(non_diag != 0, 1.0 / non_diag, 0)
        
        # 对非对角线元素进行归一化
        row_sum = inv_matrix[i, :].sum()
        if row_sum > 0:
            inv_matrix[i, :] /= row_sum  # 将非对角线元素归一化，确保它们的和为 1
    
    # 将对角线元素设置为 1，确保每行的和为 2
    np.fill_diagonal(inv_matrix, 1)

    return inv_matrix

