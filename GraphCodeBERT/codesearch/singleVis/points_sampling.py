from abc import ABC, abstractmethod

import numpy as np
import os
import time
import math
import json

import torch
from sklearn.cluster import KMeans

class HighDimensionalSampler:
    def __init__(self, data_provider, iteration, high_conf_threshold=0.8, low_conf_threshold=0.6, num_clusters=1, num_points=160000):
        self.data_provider = data_provider
        self.iteration = iteration
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.num_clusters = num_clusters
        self.num_points = num_points
    
    def sample_points(self):
        # Step 1: 读取训练数据的高维表示
        feature_vectors = self.data_provider.train_representation(self.iteration)  # (num_samples, hidden_dim)
        labels = self.data_provider.train_labels(self.iteration)  # 获取每个样本的标签
        logits = self.data_provider.get_pred(self.iteration, feature_vectors)  # 获取logits
        # 将 numpy ndarray 转换为 PyTorch Tensor
        logits_tensor = torch.from_numpy(logits)
        # 使用 softmax 计算置信度
        confidences = torch.softmax(logits_tensor, dim=1)
        predicted_classes = torch.argmax(logits_tensor, dim=1).numpy()  # 预测的类别
        
        # Step 2: 总结类别模式，定义高维空间范围
        class_centers = summarize_patterns(feature_vectors, labels, self.num_clusters)
        
        # Step 3: 计算类间低置信度点比例
        low_conf_proportion = calculate_low_conf_proportion(confidences, predicted_classes, threshold=self.low_conf_threshold)
        
        # Step 4: 在高维空间中生成高置信度和低置信度点
        high_conf_points = generate_high_conf_points(class_centers, num_points=self.num_points * 0.1)
        low_conf_points = generate_low_conf_points_with_proportion(class_centers, low_conf_proportion, num_points=self.num_points * 0.3)
        
        mid_num = self.num_points - len(high_conf_points) - len(low_conf_points)
        # Step 5: 在高、低置信度点之间生成中等置信度点
        mid_conf_points = generate_mid_conf_points_between(high_conf_points, low_conf_points, num_points=int(mid_num))

        return high_conf_points, mid_conf_points, low_conf_points
    
    # def sample_points(self):
    #     # Step 1: 读取训练数据的高维表示
    #     feature_vectors = self.data_provider.train_representation(self.iteration)  # (num_samples, num_features)
        
    #     # Step 2: 获取预测语义（置信度和类别预测）
    #     logits = self.data_provider.get_pred(self.iteration, feature_vectors)  # (num_samples, num_classes)
    #     # 将 numpy ndarray 转换为 PyTorch Tensor
    #     logits_tensor = torch.from_numpy(logits)

    #     # 使用 softmax 计算置信度
    #     confidences = torch.softmax(logits_tensor, dim=1)
    #     max_confidences, predicted_classes = torch.max(confidences, dim=1)
        
    #     # Step 3: 计算类别相似性矩阵
    #     num_classes = confidences.shape[1]
    #     similarity_matrix = compute_class_similarity(confidences, num_classes)

    #     # Step 4: 分类高、中、低置信度点
    #     high_conf_points = feature_vectors[max_confidences >= self.high_conf_threshold]
    #     mid_conf_points = feature_vectors[(max_confidences > self.low_conf_threshold) & (max_confidences < self.high_conf_threshold)]
        
    #     # Step 5: 自动采样边界点
    #     low_conf_points = sample_boundary_points(feature_vectors, confidences, predicted_classes, similarity_matrix, low_conf_threshold=self.low_conf_threshold)
        
    #     # 返回高、中、低置信度的点
    #     return high_conf_points, mid_conf_points, low_conf_points

def calculate_low_conf_proportion(confidences, predicted_classes, threshold=0.6):
    """
    计算每对类别之间的低置信度点的比例
    :param confidences: 每个样本的置信度 (num_samples, num_classes)
    :param predicted_classes: 每个样本的预测类别 (num_samples,)
    :param threshold: 置信度低于此值的点被视为低置信度
    :return: 每对类之间的低置信度点比例
    """
    num_classes = confidences.shape[1]
    low_conf_matrix = np.zeros((num_classes, num_classes))

    # 遍历所有样本，统计每对类别的低置信度点数量
    for i in range(confidences.shape[0]):
        pred_class = predicted_classes[i]
        # 找到第二高的置信度的类别
        sorted_conf_indices = np.argsort(-confidences[i])
        second_class = sorted_conf_indices[1]  # 第二高的置信度类别

        # 如果两个类别的置信度都低于阈值，则认为它们之间存在低置信度点
        if confidences[i, pred_class] < threshold and confidences[i, second_class] < threshold:
            low_conf_matrix[pred_class, second_class] += 1
            low_conf_matrix[second_class, pred_class] += 1  # 对称关系
    
    # 将低置信度点的数量转换为比例
    total_low_conf = np.sum(low_conf_matrix)
    if total_low_conf > 0:
        low_conf_proportion = low_conf_matrix / total_low_conf
    else:
        low_conf_proportion = low_conf_matrix  # 如果没有低置信度点，则保持为0
    
    return low_conf_proportion

    
def summarize_patterns(feature_vectors, labels, num_clusters=1):
    """
    对每个类别进行聚类，总结高维空间中的模式
    :param feature_vectors: 训练数据的高维表示 (num_samples, hidden_dim)
    :param labels: 每个样本的类别标签 (num_samples,)
    :param num_clusters: 每个类别的聚类数量
    :return: 每个类别的聚类中心
    """
    class_centers = {}
    for cls in np.unique(labels):
        cls_data = feature_vectors[labels == cls]
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(cls_data)
        class_centers[cls] = kmeans.cluster_centers_  # 聚类中心
    return class_centers

def generate_high_conf_points(class_centers, num_points=1000):
    """
    在类别中心附近生成高置信度点
    :param class_centers: 类别的聚类中心
    :param num_points: 每个类别生成的点数
    :return: 高置信度点
    """
    high_conf_points = []
    for cls, centers in class_centers.items():
        for center in centers:
            # 在类别中心附近添加扰动
            points = center + np.random.normal(scale=0.05, size=(int(num_points) // len(class_centers), center.shape[0]))
            high_conf_points.append(points)
    return np.vstack(high_conf_points)

def generate_mid_conf_points_between(high_conf_points, low_conf_points, num_points=1000):
    """
    在高置信度和低置信度点之间生成中等置信度点
    :param high_conf_points: 高置信度点
    :param low_conf_points: 低置信度点
    :param num_points: 生成的中等置信度点的数量
    :return: 中等置信度点
    """
    assert high_conf_points.shape[1] == low_conf_points.shape[1], "High and Low confidence points must have the same dimensionality."
    
    mid_conf_points = []
    # 随机从高置信度和低置信度点中选择点配对
    for _ in range(num_points):
        idx_high = np.random.randint(0, high_conf_points.shape[0])
        idx_low = np.random.randint(0, low_conf_points.shape[0])
        
        # 获取一对高置信度点和低置信度点
        high_point = high_conf_points[idx_high]
        low_point = low_conf_points[idx_low]
        
        # 随机生成插值比例 alpha
        alpha = np.random.uniform(0.3, 0.7)  # 保持在高、低置信度点之间的适中位置
        
        # 生成中等置信度点，通过高低置信度点的线性插值
        mid_point = alpha * high_point + (1 - alpha) * low_point
        mid_conf_points.append(mid_point)
    
    return np.vstack(mid_conf_points)

def generate_low_conf_points_with_proportion(class_centers, low_conf_proportion, num_points=1000, boundary_factor=0.5):
    """
    按照类间低置信度比例生成低置信度点
    :param class_centers: 类别的聚类中心
    :param low_conf_proportion: 类间低置信度点的比例矩阵
    :param num_points: 总共生成的低置信度点数量
    :param boundary_factor: 生成低置信度点的扰动因子
    :return: 低置信度点
    """
    low_conf_points = []
    classes = list(class_centers.keys())
    
    # 计算每对类生成的低置信度点数量
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            proportion = low_conf_proportion[classes[i], classes[j]]
            num_to_generate = int(proportion * num_points)
            
            if num_to_generate > 0:
                cls_i_centers = class_centers[classes[i]]
                cls_j_centers = class_centers[classes[j]]
                
                # 生成该类别对之间的低置信度点
                for center_i in cls_i_centers:
                    for center_j in cls_j_centers:
                        boundary_center = (center_i + center_j) / 2
                        points = boundary_center + np.random.normal(scale=boundary_factor, size=(num_to_generate // len(cls_i_centers), center_i.shape[0]))
                        low_conf_points.append(points)
    
    return np.vstack(low_conf_points)



def generate_low_conf_points(class_centers, num_points=1000, boundary_factor=0.5):
    """
    在类别之间的边界生成低置信度点
    :param class_centers: 类别的聚类中心
    :param num_points: 每个类别生成的点数
    :param boundary_factor: 边界点生成的扰动因子
    :return: 低置信度点
    """
    low_conf_points = []
    classes = list(class_centers.keys())
    
    # 遍历所有类别对，生成边界点
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            cls_i_centers = class_centers[classes[i]]
            cls_j_centers = class_centers[classes[j]]
            
            # 计算类别i和类别j之间的边界中心
            for center_i in cls_i_centers:
                for center_j in cls_j_centers:
                    boundary_center = (center_i + center_j) / 2
                    # 在边界中心附近生成低置信度点
                    points = boundary_center + np.random.normal(scale=boundary_factor, size=(num_points // len(cls_i_centers), center_i.shape[0]))
                    low_conf_points.append(points)
    
    return np.vstack(low_conf_points)


def compute_class_similarity(confidences, num_classes):
    """
    计算类别之间的相似性
    :param confidences: 每个样本的预测置信度 (num_samples, num_classes)
    :param num_classes: 类别的数量
    :return: 类别之间的相似性矩阵 (num_classes, num_classes)
    """
    # 初始化类别相似性矩阵
    class_similarity = torch.zeros((num_classes, num_classes))

    # 对于每对类别，计算它们的相似性
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            # 计算类别i和类别j的置信度差异
            confidence_diff = torch.abs(confidences[:, i] - confidences[:, j])
            # 相似性可以定义为1减去平均差异
            class_similarity[i, j] = 1 - torch.mean(confidence_diff)
            class_similarity[j, i] = class_similarity[i, j]
    
    return class_similarity


def sample_boundary_points(feature_vectors, confidences, predicted_classes, similarity_matrix, low_conf_threshold=0.6):
    """
    采样类之间的边界点，基于类间的相似性自动调整采样权重
    :param feature_vectors: 样本的高维表示 (num_samples, hidden_dim)
    :param confidences: 每个样本的置信度 (num_samples, num_classes)
    :param predicted_classes: 每个样本的预测类别 (num_samples,)
    :param similarity_matrix: 类别相似性矩阵 (num_classes, num_classes)
    :param low_conf_threshold: 低置信度阈值
    :return: 类别间的边界点
    """
    boundary_points = []
    num_classes = similarity_matrix.shape[0]
    
    # 遍历所有类别对
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            # 获取类别i和类别j之间的相似性
            similarity = similarity_matrix[i, j]
            
            # 根据相似性设置采样权重，相似性越高，边界点采样的概率越大
            sampling_weight = similarity

            # 找到类别i和类别j之间的边界点（置信度差异小且置信度较低）
            boundary_mask = (torch.abs(confidences[:, i] - confidences[:, j]) < 0.1) & \
                            (confidences[:, i] <= low_conf_threshold) & \
                            (confidences[:, j] <= low_conf_threshold)

            # 根据采样权重随机采样边界点
            boundary_indices = torch.nonzero(boundary_mask).squeeze()
            sampled_boundary_points = feature_vectors[boundary_indices]

            # 将采样的 numpy array 转换为 Tensor
            if isinstance(sampled_boundary_points, np.ndarray):
                sampled_boundary_points = torch.from_numpy(sampled_boundary_points)

            num_to_sample = int(sampling_weight * len(sampled_boundary_points))

            if num_to_sample > 0:
                sampled_boundary_points = sampled_boundary_points[:num_to_sample]
                boundary_points.append(sampled_boundary_points)
    
    # 确保 boundary_points 是一个 Tensor 的列表
    return torch.cat(boundary_points, dim=0)

