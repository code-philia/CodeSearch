import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_most_similar_sample(training_data, training_predictions, target_prob_dists):
    """
    在训练数据中找到预测语义最接近目标概率分布的样本
    :param training_data: 训练数据的高维嵌入 (num_samples, 512)
    :param training_predictions: 训练数据的预测语义 (num_samples, 10)
    :param target_prob_dists: 目标多个10维预测语义 (num_targets, 10)
    :return: 每个目标语义对应的最相似样本的索引列表和相应的样本
    """
    
    most_similar_idxs = []
    most_similar_samples = []
    
    # 对于每个目标预测语义 (target_prob_dist 是每个 10 维向量)
    for target_prob_dist in target_prob_dists:
        # 计算训练数据的预测结果与该目标概率分布的相似度（余弦相似度）
        similarities = cosine_similarity(training_predictions, target_prob_dist.reshape(1, -1))
        
        # 找到与该目标最相似的训练样本索引
        most_similar_idx = similarities.argmax()
        
        # 记录该索引及相应的样本
        most_similar_idxs.append(most_similar_idx)
        most_similar_samples.append(training_data[most_similar_idx])
    
    return most_similar_idxs, most_similar_samples

def perturb_sample(sample, target_prob_dist, data_provider, epoch, DEVICE, lr=0.01, num_steps=100):
    """
    针对最相似样本进行扰动或优化，使其预测语义逼近目标概率分布
    :param sample: 最相似的样本 (512维高维嵌入)
    :param target_prob_dist: 目标10维预测语义
    :param data_provider: 用于数据处理
    :param lr: 学习率
    :param num_steps: 优化的步数
    :return: 优化后的样本
    """
    # 将样本设置为可训练
    sample = torch.tensor(sample, dtype=torch.float32).clone().detach().requires_grad_(True)
    
    # 定义优化器
    optimizer = optim.Adam([sample], lr=lr)
    
    # 将目标概率分布转为Tensor
    target_prob_dist = torch.tensor(target_prob_dist, dtype=torch.float32).to(DEVICE)

    # 优化过程
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # 计算扰动后样本的预测语义
        predicted_prob_dist = data_provider.get_pred_tensor(epoch, sample)
        # 如果 get_pred 返回的是 NumPy 数组，使用 from_numpy 转换为 PyTorch Tensor
        # if isinstance(predicted_prob_dist, np.ndarray):
        #     predicted_prob_dist = torch.from_numpy(predicted_prob_dist).float()

        # # 确保 predicted_prob_dist 仍然在计算图中
        # predicted_prob_dist = predicted_prob_dist.requires_grad_(True)

        # 计算损失，目标是使预测语义接近目标概率分布
        loss = F.mse_loss(predicted_prob_dist, target_prob_dist)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印优化过程中的损失
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item()}")

    return sample.detach()