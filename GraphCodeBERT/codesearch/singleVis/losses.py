from abc import ABC, abstractmethod
import torch
from torch import nn
from singleVis.backend import convert_distance_to_probability, compute_cross_entropy
from scipy.special import softmax
import torch.nn.functional as F
import torch.optim as optim
import os
import copy


        
import torch
torch.manual_seed(0)  # fixed seed
torch.cuda.manual_seed_all(0)
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

import json
from datetime import datetime
# Set the random seed for numpy

"""Losses modules for preserving four propertes"""
# https://github.com/ynjnpa/VocGAN/blob/5339ee1d46b8337205bec5e921897de30a9211a1/utils/stft_loss.py for losses module

class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class MyModel(nn.Module):
    def __init__(self, initial_tensor):
        super(MyModel, self).__init__()
        self.learnable_matrix = nn.Parameter(initial_tensor.clone().detach())

class UmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device,  data_provider, epoch, net, fixed_number = 5, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)
        self.data_provider = data_provider
        self.epoch = epoch
        self.net = net
        self.model_path = os.path.join(self.data_provider.content_path, "Model")
        self.fixed_number = fixed_number

        model_location = os.path.join(self.model_path, "{}_{:d}".format('Epoch', epoch), "subject_model.pth")
        self.net.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")),strict=False)
        self.net.to(self.DEVICE)
        self.net.train()

        for param in net.parameters():
            param.requires_grad = False

        self.pred_fn = self.net.prediction

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, edge_to_idx, edge_from_idx,embedding_to, embedding_from, probs, pred_edge_to, pred_edge_from,edge_to, edge_from,recon_to, recon_from,a_to, a_from,recon_pred_edge_to,recon_pred_edge_from,curr_model,iteration, data_provider,epoch, points_2d):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        pred_edge_to_neg_Res = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        pred_repeat_neg = torch.repeat_interleave(pred_edge_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]
        pred_edge_from_neg_Res = pred_repeat_neg[randperm]
        indicates = self.filter_neg(pred_edge_from_neg_Res, pred_edge_to_neg_Res)

        #### strategy confidence: filter negative
        embedding_neg_to = embedding_neg_to[indicates]
        embedding_neg_from = embedding_neg_from[indicates]

        neg_num = len(embedding_neg_from)

        positive_distance = torch.norm(embedding_to - embedding_from, dim=1)
        negative_distance = torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
        #  distances between samples (and negative samples)
        positive_distance_mean = torch.mean(positive_distance)
        negative_distance_mean = torch.mean(negative_distance)

        #### dynamic labeling
        pred_edge_to_Res = pred_edge_to.argmax(axis=1)
        pred_edge_from_Res = pred_edge_from.argmax(axis=1)

        is_pred_same = (pred_edge_to_Res.to(self.DEVICE) == pred_edge_from_Res.to(self.DEVICE))
        is_pred_same = is_pred_same.to(self.DEVICE)
        pred_edge_to = pred_edge_to.to(self.DEVICE)
        pred_edge_from = pred_edge_from.to(self.DEVICE)

        recon_pred_to_Res = recon_pred_edge_to.argmax(axis=1)
        recon_pred_from_Res = recon_pred_edge_from.argmax(axis=1)

        is_pred_correct = torch.logical_and(
            pred_edge_to_Res.to(self.DEVICE) == recon_pred_to_Res.to(self.DEVICE),
            pred_edge_from_Res.to(self.DEVICE) == recon_pred_from_Res.to(self.DEVICE)
        )

        temp = 0.001
        recon_pred_to_softmax = F.softmax(recon_pred_edge_to / temp, dim=-1)
        recon_pred_from_softmax = F.softmax(recon_pred_edge_from / temp, dim=-1)

        pred_to_softmax = F.softmax(pred_edge_to / temp, dim=-1)
        pred_from_softmax = F.softmax(pred_edge_from / temp, dim=-1)
        
        recon_pred_to_softmax = torch.Tensor(recon_pred_to_softmax.to(self.DEVICE))
        recon_pred_from_softmax = torch.Tensor(recon_pred_from_softmax.to(self.DEVICE))
        #### umap loss
        distance_embedding = torch.cat(
            (
                positive_distance,
                negative_distance,
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        probabilities_graph = torch.cat(
            (probs, torch.zeros(neg_num).to(self.DEVICE)), dim=0,
        )

        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )  


        batch_margin = positive_distance_mean +  (negative_distance_mean - positive_distance_mean) * (1-probs)
        init_margin = (1.0 - is_pred_same.float()) * batch_margin

        if iteration > self.fixed_number:
            pred_edge_to_prob = compute_pred_confidence_diff(pred_edge_to)
            pred_edge_from_prob = compute_pred_confidence_diff(pred_edge_from)
            recon_pred_edge_to_prob = compute_pred_confidence_diff(recon_pred_edge_to)
            recon_pred_edge_from_prob = compute_pred_confidence_diff(recon_pred_edge_from)
            # np.save('/home/yiming/cophi/projects/Trustvis/debugging_observation/is_pred_same.npy', is_pred_same.cpu().numpy())
            is_conf_same = find_same_confidence(pred_edge_to_prob, pred_edge_from_prob, threshold=0.1)
            is_conf_recon_same = find_same_confidence_recon(pred_edge_to_prob, pred_edge_from_prob, recon_pred_edge_to_prob, recon_pred_edge_from_prob, interval=0.1)

            pred_same_not_conf_same = is_pred_same & (~is_conf_same)
            # pred_same_not_conf_same = is_pred_same & (~is_conf_same) & (is_pred_correct)
            embedding_to = embedding_to.to(self.DEVICE)
            embedding_from = embedding_from.to(self.DEVICE)
            points_2d = points_2d.to(self.DEVICE)
            emb_to_1, emb_from_1 = self.find_best_embedding_via_projection_local(embedding_to[pred_same_not_conf_same],embedding_from[pred_same_not_conf_same], pred_edge_to[pred_same_not_conf_same], pred_edge_from[pred_same_not_conf_same], points_2d, curr_model, data_provider, epoch, radius=0.5)
            # 计算最终的 margin
            # print("emb_to_1", emb_to_1)
            # print("emb_to", embedding_to[pred_same_not_conf_same])
            # print("emb_from_1", emb_from_1)
            # print("emb_from", embedding_from[pred_same_not_conf_same])
            confidence_margin = self.compute_final_margin(embedding_to[pred_same_not_conf_same],embedding_from[pred_same_not_conf_same], emb_to_1, emb_from_1)
            # print("avg conf margin:", confidence_margin.mean())
            # 置信度差异样本之间的 margin，直接计入损失
            # conf_loss = (confidence_margin.to(self.DEVICE)).mean()
            # 最小-最大归一化
            # normalized_confidence_margin = (confidence_margin - torch.min(confidence_margin)) / (torch.max(confidence_margin) - torch.min(confidence_margin))

            # scale_dist = torch.min(negative_distance)
            # scaled_confidence_margin = normalized_confidence_margin * scale_dist
            # print("scaled_confidence_margin", scaled_confidence_margin)
            # conf_loss = (confidence_margin.to(self.DEVICE)).sum() / batch_size
            init_margin[pred_same_not_conf_same] = confidence_margin
            # 预测结果不同样本之间的 margin，使用 ReLU 激活
            pred_loss = F.relu(init_margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()

            # 最终的 margin_loss
            margin_loss = pred_loss
        else:
            # print("avg pred margin:", batch_margin[~is_pred_same].mean())
            margin = init_margin
            margin_loss = F.relu(margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()

        # if iteration > self.fixed_number:
        #     pred_edge_to_prob = compute_pred_confidence_diff(pred_edge_to)
        #     pred_edge_from_prob = compute_pred_confidence_diff(pred_edge_from)
        #     recon_pred_edge_to_prob = compute_pred_confidence_diff(recon_pred_edge_to)
        #     recon_pred_edge_from_prob = compute_pred_confidence_diff(recon_pred_edge_from)
        #     # np.save('/home/yiming/cophi/projects/Trustvis/debugging_observation/is_pred_same.npy', is_pred_same.cpu().numpy())
        #     is_conf_same = find_same_confidence(pred_edge_to_prob, pred_edge_from_prob, interval=0.2)
        #     # is_conf_recon_same = find_same_confidence_recon(pred_edge_to_prob, pred_edge_from_prob, recon_pred_edge_to_prob, recon_pred_edge_from_prob, interval=0.2)

        #     # pred_same_not_conf_same = is_pred_same & (~is_conf_same)
        #     pred_same_not_conf_same = is_pred_same & (~is_conf_same) & (is_pred_correct)
        #     # Compute confidence difference for pred_same but not conf_same samples
        #     # _, to_indices = torch.max(pred_edge_to[pred_same_not_conf_same], dim=1)
        #     # _, from_indices = torch.max(pred_edge_from[pred_same_not_conf_same], dim=1)

        #     _, sorted_indices_to = torch.sort(pred_edge_to[pred_same_not_conf_same], dim=1, descending=True)
        #     # 第一大值的索引是每行排序后的第一个值
        #     first_indices_to = sorted_indices_to[:, 0]
        #     # 第二大值的索引是每行排序后的第二个值
        #     second_indices_to = sorted_indices_to[:, 1]

        #     _, sorted_indices_from = torch.sort(pred_edge_from[pred_same_not_conf_same], dim=1, descending=True)
        #     # 第一大值的索引是每行排序后的第一个值
        #     first_indices_from = sorted_indices_from[:, 0]
        #     # 第二大值的索引是每行排序后的第二个值
        #     second_indices_from = sorted_indices_from[:, 1]
           
        #     emb_to_1, emb_from_1 = self.compute_dynamic_margin_via_projection(embedding_to[pred_same_not_conf_same],embedding_from[pred_same_not_conf_same],curr_model, pred_edge_to_prob[pred_same_not_conf_same], pred_edge_from_prob[pred_same_not_conf_same], first_indices_to, first_indices_from, second_indices_to, second_indices_from, pred_edge_to[pred_same_not_conf_same], pred_edge_from[pred_same_not_conf_same], data_provider, epoch)
        #     # 计算最终的 margin
        #     confidence_margin = self.compute_final_margin(embedding_to[pred_same_not_conf_same],embedding_from[pred_same_not_conf_same], emb_to_1, emb_from_1)
        #     # print("avg conf margin:", confidence_margin.mean())
        #     # 置信度差异样本之间的 margin，直接计入损失
        #     # conf_loss = (confidence_margin.to(self.DEVICE)).mean()
        #     # 最小-最大归一化
        #     normalized_confidence_margin = (confidence_margin - torch.min(confidence_margin)) / (torch.max(confidence_margin) - torch.min(confidence_margin))

        #     scaled_confidence_margin = normalized_confidence_margin * torch.min(negative_distance)
        #     # print("scaled_confidence_margin", scaled_confidence_margin)
        #     conf_loss = (scaled_confidence_margin.to(self.DEVICE)).sum() / batch_size
        #     # 预测结果不同样本之间的 margin，使用 ReLU 激活
        #     pred_loss = F.relu(init_margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()

        #     # 最终的 margin_loss
        #     margin_loss = pred_loss + conf_loss
        # else:
        #     # print("avg pred margin:", batch_margin[~is_pred_same].mean())
        #     margin = init_margin
        #     margin_loss = F.relu(margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()
        
        umap_l = torch.mean(ce_loss).to(self.DEVICE) 
        margin_loss = margin_loss.to(self.DEVICE)

        if torch.isnan(margin_loss):
            margin_loss = torch.tensor(0.0).to(margin_loss.device)

        return umap_l, margin_loss, umap_l+margin_loss
    
    # def forward(self, edge_to_idx, edge_from_idx,embedding_to, embedding_from, probs, pred_edge_to, pred_edge_from,edge_to, edge_from,recon_to, recon_from,a_to, a_from,recon_pred_edge_to,recon_pred_edge_from,curr_model,iteration):
    #     batch_size = embedding_to.shape[0]
    #     # get negative samples
    #     embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
    #     pred_edge_to_neg_Res = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
    #     repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
    #     pred_repeat_neg = torch.repeat_interleave(pred_edge_from, self._negative_sample_rate, dim=0)
    #     randperm = torch.randperm(repeat_neg.shape[0])
    #     embedding_neg_from = repeat_neg[randperm]
    #     pred_edge_from_neg_Res = pred_repeat_neg[randperm]
    #     indicates = self.filter_neg(pred_edge_from_neg_Res, pred_edge_to_neg_Res)

    #     #### strategy confidence: filter negative
    #     embedding_neg_to = embedding_neg_to[indicates]
    #     embedding_neg_from = embedding_neg_from[indicates]

    #     neg_num = len(embedding_neg_from)

    #     positive_distance = torch.norm(embedding_to - embedding_from, dim=1)
    #     negative_distance = torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
    #     #  distances between samples (and negative samples)
    #     positive_distance_mean = torch.mean(positive_distance)
    #     negative_distance_mean = torch.mean(negative_distance)

    #     #### dynamic labeling
    #     pred_edge_to_Res = pred_edge_to.argmax(axis=1)
    #     pred_edge_from_Res = pred_edge_from.argmax(axis=1)

    #     is_pred_same = (pred_edge_to_Res.to(self.DEVICE) == pred_edge_from_Res.to(self.DEVICE))
    #     is_pred_same = is_pred_same.to(self.DEVICE)
    #     pred_edge_to = pred_edge_to.to(self.DEVICE)
    #     pred_edge_from = pred_edge_from.to(self.DEVICE)

    #     recon_pred_to_Res = recon_pred_edge_to.argmax(axis=1)
    #     recon_pred_from_Res = recon_pred_edge_from.argmax(axis=1)


    #     temp = 0.001
    #     recon_pred_to_softmax = F.softmax(recon_pred_edge_to / temp, dim=-1)
    #     recon_pred_from_softmax = F.softmax(recon_pred_edge_from / temp, dim=-1)

    #     pred_to_softmax = F.softmax(pred_edge_to / temp, dim=-1)
    #     pred_from_softmax = F.softmax(pred_edge_from / temp, dim=-1)
        
    #     recon_pred_to_softmax = torch.Tensor(recon_pred_to_softmax.to(self.DEVICE))
    #     recon_pred_from_softmax = torch.Tensor(recon_pred_from_softmax.to(self.DEVICE))
    #     #### umap loss
    #     distance_embedding = torch.cat(
    #         (
    #             positive_distance,
    #             negative_distance,
    #         ),
    #         dim=0,
    #     )
    #     probabilities_distance = convert_distance_to_probability(
    #         distance_embedding, self.a, self.b
    #     )
    #     probabilities_distance = probabilities_distance.to(self.DEVICE)

    #     probabilities_graph = torch.cat(
    #         (probs, torch.zeros(neg_num).to(self.DEVICE)), dim=0,
    #     )

    #     probabilities_graph = probabilities_graph.to(device=self.DEVICE)

    #     # compute cross entropy
    #     (_, _, ce_loss) = compute_cross_entropy(
    #         probabilities_graph,
    #         probabilities_distance,
    #         repulsion_strength=self._repulsion_strength,
    #     )  


    #     batch_margin = positive_distance_mean +  (negative_distance_mean - positive_distance_mean) * (1-probs)
    #     init_margin = (1.0 - is_pred_same.float()) * batch_margin

    #     if iteration > self.fixed_number:
    #         margin = self.newton_step_with_regularization(edge_to_idx, edge_from_idx,init_margin, is_pred_same, 
    #                                                   edge_to[~is_pred_same],edge_from[~is_pred_same], probs[~is_pred_same],
    #                                                   embedding_to[~is_pred_same],embedding_from[~is_pred_same],curr_model,
    #                                                   pred_to_softmax[~is_pred_same], pred_from_softmax[~is_pred_same],positive_distance_mean,negative_distance_mean)
    #     else:
    #         margin = init_margin
        
    #     # print(margin[~is_pred_same].mean().item(),positive_distance.mean().item(), positive_distance[~is_pred_same].mean().item())
        
    #     # print("dynamic marin", margin[~is_pred_same].mean())
    #     # print("margin", margin.mean().item())
    #     margin_loss = F.relu(margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()
        
    #     umap_l = torch.mean(ce_loss).to(self.DEVICE) 
    #     margin_loss = margin_loss.to(self.DEVICE)

    #     if torch.isnan(margin_loss):
    #         margin_loss = torch.tensor(0.0).to(margin_loss.device)

    #     return umap_l, margin_loss, umap_l+margin_loss

    def compute_final_margin(self, emb_to, emb_from, emb_to_1, emb_from_1):
        # 计算 emb_to_1 和 emb_from 之间的距离 d1
        # d1 = torch.norm(emb_to_1 - emb_from, dim=1)
        # # print("d1", d1)
        
        # # 计算 emb_from_1 和 emb_to 之间的距离 d2
        # d2 = torch.norm(emb_from_1 - emb_to, dim=1)
        # # print("d2", d2)
        
        # # 取 d1 和 d2 的最大值
        # final_margin = torch.max(d1, d2)
        # print("final_margin", final_margin)
        final_margin = torch.norm(emb_to_1 - emb_from_1, dim=1)
        
        return final_margin

    def compute_dynamic_margin_via_projection(self, emb_to, emb_from, curr_model, pred_to_conf, pred_from_conf, first_indices_to, first_indices_from, second_indices_to, second_indices_from, pred_edge_to, pred_edge_from, data_provider,epoch, max_iterations=40, lr=0.05):
        # 初始化 emb_to_1 和 emb_from_1
        emb_to_1 = emb_to.clone().detach().requires_grad_(True)
        emb_from_1 = emb_from.clone().detach().requires_grad_(True)
        # combined_emb = torch.cat([emb_to_1, emb_from_1], dim=0)
        combined_indices = torch.cat([first_indices_to, first_indices_from], dim=0)
        combined_indices_second = torch.cat([second_indices_to, second_indices_from], dim=0)
        combined_truth_conf = torch.cat([pred_to_conf, pred_from_conf], dim=0)
        combined_truth_pred = torch.cat([pred_edge_to, pred_edge_from], dim=0)
        
        optimizer = torch.optim.Adam([emb_to_1, emb_from_1], lr=lr)
        early_stop_threshold = 0.2  # 设置 early stopping 的阈值
        min_loss_threshold = 1e-3  # 最小 loss 阈值，用于停止优化
        # 使用深度拷贝来确保模型不共享计算图
        # curr_model_to = copy.deepcopy(curr_model)
        # curr_model_from = copy.deepcopy(curr_model)
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            # 将 emb_to_1 和 emb_from_1 连接起来
            combined_emb = torch.cat([emb_to_1, emb_from_1], dim=0)

            # 传递给 decoder
            combined_decoder_output = curr_model.decoder(combined_emb)
            combined_pred = data_provider.get_pred_tensor(epoch, combined_decoder_output).to(device=self.DEVICE, dtype=torch.float32)

            loss_fn = torch.nn.MSELoss()
            # pred_loss = loss_fn(combined_pred, combined_truth_pred)

            # print("combined_pred", combined_pred[:5])
            # print("combined_truth_pred", combined_truth_pred[:5])
            # print("pred_loss", pred_loss)

            # pred_to_1, pred_from_1 = torch.split(combined_pred, emb_to_1.shape[0], dim=0)
            # print("before conf:", combined_pred.requires_grad, combined_pred.grad_fn, combined_pred._version)
            # 通过 decoder 获取新的预测
            combined_conf = compute_recon_pred_confidence_diff(combined_pred, combined_indices, combined_indices_second)

            _, conf_indices = torch.max(combined_pred, dim=1)
            # print("conf_indices", conf_indices[:5])
            # print("combined_indices", combined_indices[:5])

            # print("after conf:", combined_conf.requires_grad, combined_conf.grad_fn, combined_conf._version)
            # 假设 combined_conf 和 combined_truth_conf 是 (1400, 2) 的张量
            # half_size = combined_conf.size(0) // 2

            # # 将前一半和后一半分别取出
            # conf_first_half = combined_conf[:half_size]
            # conf_second_half = combined_conf[half_size:]

            # truth_conf_first_half = combined_truth_conf[:half_size]
            # truth_conf_second_half = combined_truth_conf[half_size:]

            # print("combined_conf", combined_conf[:5])
            # print("combined_truth_conf", combined_truth_conf[:5])
            diff_loss = loss_fn(combined_conf, combined_truth_conf)
            # print("diff_loss", diff_loss)
            # 分别进行相减操作
            # first_half_diff = torch.mean((conf_first_half - truth_conf_first_half) ** 2)
            # second_half_diff = torch.mean((conf_second_half - truth_conf_second_half) ** 2)
            total_loss = diff_loss
            # print("first_half_diff", first_half_diff)
            # print("second_half_diff", second_half_diff)
            # total_loss = torch.mean((combined_conf - combined_truth_conf) ** 2)

            # conf_to_1 = compute_recon_pred_confidence_diff(combined_pred[0], to_indices)
            # print("after conf:", conf_to_1.requires_grad, conf_to_1.grad_fn, conf_to_1._version)
            # # 损失函数，确保 emb_to_1 和 emb_from_1 落在目标 confidence 区间内
            # loss_to_1 = torch.mean((conf_to_1 - pred_to_conf) ** 2)

            # # print("before conf:", pred_from_1.requires_grad, pred_from_1.grad_fn, pred_from_1._version)
            # conf_from_1 = compute_recon_pred_confidence_diff(combined_pred[1], from_indices)
            # print("after conf:", conf_from_1.requires_grad, conf_from_1.grad_fn, conf_from_1._version)

            # loss_from_1 = torch.mean((conf_from_1 - pred_from_conf) ** 2)
            # total_loss = first_half_diff + second_half_diff
            # print("before back:", total_loss.requires_grad, total_loss.grad_fn)

            # 检查 early stopping 条件
            # max_conf_diff = max(torch.mean(torch.abs(conf_to_1 - pred_to_conf)).item(), torch.mean(torch.abs(conf_from_1 - pred_from_conf)).item())
            max_conf_diff = torch.mean(torch.abs(combined_conf - combined_truth_conf)).item()

            if max_conf_diff < early_stop_threshold and total_loss.item() < min_loss_threshold:
                print(f"Early stopping at iteration {iteration}, max_conf_diff: {max_conf_diff}, total_loss: {total_loss.item()}")
                break
            # print("before back:", total_loss.requires_grad, total_loss.grad_fn, total_loss._version)
            # print("total_loss", total_loss)
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

        return emb_to_1, emb_from_1
    
    def find_best_embedding_via_projection(self, ground_truth_to, ground_truth_from, points_2d, curr_model, data_provider, epoch):
        """
        对每个 ground truth 的语义，找到二维平面上最相似的点作为 emb_to_1 和 emb_from_1。

        Parameters:
        ground_truth_to (torch.Tensor): ground truth 对应的 'to' 语义。
        ground_truth_from (torch.Tensor): ground truth 对应的 'from' 语义。
        points_2d (torch.Tensor): 二维平面上的候选点 (N, 2)。
        curr_model (nn.Module): 当前模型，用于 decoder 操作。
        data_provider (DataProvider): 用于获取预测张量。
        epoch (int): 当前的训练 epoch。

        Returns:
        emb_to_1 (torch.Tensor): 最相似的 'to' 点嵌入。
        emb_from_1 (torch.Tensor): 最相似的 'from' 点嵌入。
        """
        
        # 将所有二维点传入 decoder 进行预测
        points_2d = points_2d.to(self.DEVICE)
        decoder_output = curr_model.decoder(points_2d)

        # 使用 get_pred_tensor 获取语义预测结果
        predicted_semantics = data_provider.get_pred_tensor(epoch, decoder_output).to(device=self.DEVICE, dtype=torch.float32)
        # 扩展 predicted_semantics 的第 1 维，变成 (40000, 1, 10)
        predicted_semantics = predicted_semantics.unsqueeze(1)  # 变为 (40000, 1, 10)

        ground_truth_to = ground_truth_to.unsqueeze(0)
        similarities_to = torch.nn.functional.cosine_similarity(predicted_semantics, ground_truth_to, dim=2)

        ground_truth_from = ground_truth_from.unsqueeze(0) 
        similarities_from = torch.nn.functional.cosine_similarity(predicted_semantics, ground_truth_from, dim=2)

        # 找到与 ground_truth_to 和 ground_truth_from 最相似的点
        best_to_idx = torch.argmax(similarities_to, dim=0)
        best_from_idx = torch.argmax(similarities_from, dim=0)

        # 获取最相似点的二维嵌入坐标
        emb_to_1 = points_2d[best_to_idx]
        emb_from_1 = points_2d[best_from_idx]

        return emb_to_1, emb_from_1

    def find_best_embedding_via_projection_local(self, emb_to, emb_from, ground_truth_to, ground_truth_from, points_2d, curr_model, data_provider, epoch, radius=0.1):
        """
        对每个 ground truth 的语义，只在 emb_to 和 emb_from 附近的点中寻找最相似的点作为 emb_to_1 和 emb_from_1。

        Parameters:
        emb_to (torch.Tensor): 当前投影位置的 emb_to。
        emb_from (torch.Tensor): 当前投影位置的 emb_from。
        ground_truth_to (torch.Tensor): ground truth 对应的 'to' 语义。
        ground_truth_from (torch.Tensor): ground truth 对应的 'from' 语义。
        points_2d (torch.Tensor): 二维平面上的候选点 (N, 2)。
        curr_model (nn.Module): 当前模型，用于 decoder 操作。
        data_provider (DataProvider): 用于获取预测张量。
        epoch (int): 当前的训练 epoch。
        radius (float): 定义投影点的邻域半径。

        Returns:
        emb_to_1 (torch.Tensor): 最相似的 'to' 点嵌入。
        emb_from_1 (torch.Tensor): 最相似的 'from' 点嵌入。
        """

        # 使用 get_pred_tensor 获取语义预测结果
        decoder_output = curr_model.decoder(points_2d)
        predicted_semantics = data_provider.get_pred_tensor(epoch, decoder_output).to(device=self.DEVICE, dtype=torch.float32)

        # 扩展 emb_to 的维度，使其变为 (x, 1, 2)，然后使用广播机制与 (40000, 2) 进行距离计算
        distance_to_emb_to = torch.norm(points_2d.unsqueeze(1) - emb_to.unsqueeze(0), dim=2)
        # 对 emb_from 执行同样的操作
        distance_to_emb_from = torch.norm(points_2d.unsqueeze(1) - emb_from.unsqueeze(0), dim=2)

        # 找到距离 emb_to 和 emb_from 小于 radius 的点（邻域内的点）
        nearby_points_to_mask = distance_to_emb_to < radius
        nearby_points_from_mask = distance_to_emb_from < radius

        sim_fn = torch.nn.functional.cosine_similarity  # 定义相似度函数
    
        # 初始化存储最相似点的列表
        emb_to_1_list = []
        emb_from_1_list = []
        
        # Step 4: 对于每个 emb_to 和 emb_from，计算其邻域内的点的相似度
        for i in range(emb_to.size(0)):  # 遍历 779 个 emb_to
            # 提取当前 emb_to 的邻域内点的 mask
            valid_points_mask_to = nearby_points_to_mask[:, i]
            valid_points_mask_from = nearby_points_from_mask[:, i]

            # 提取邻域内的 points_2d 和相应的 predicted_semantics
            nearby_points_to = points_2d[valid_points_mask_to]
            nearby_semantics_to = predicted_semantics[valid_points_mask_to]

            nearby_points_from = points_2d[valid_points_mask_from]
            nearby_semantics_from = predicted_semantics[valid_points_mask_from]
            
            # Step 5: 处理异常情况
            # 如果没有满足条件的点，将 emb_to_1 和 emb_from_1 设置为原始的 emb_to 和 emb_from
            if nearby_points_to.size(0) == 0:
                # print(f"No valid points found for emb_to[{i}], setting emb_to_1 as original emb_to[{i}].")
                emb_to_1_list.append(emb_to[i])
            else:
                # 计算与 ground_truth_to[i] 的相似度
                similarities_to = sim_fn(nearby_semantics_to, ground_truth_to[i].unsqueeze(0).expand_as(nearby_semantics_to), dim=1)

                if similarities_to.numel() > 0:
                    best_to_idx = torch.argmax(similarities_to)
                    emb_to_1 = nearby_points_to[best_to_idx]
                    emb_to_1_list.append(emb_to_1)
                else:
                    # print(f"No valid similarity values for emb_to[{i}], setting emb_to_1 as original emb_to[{i}].")
                    emb_to_1_list.append(emb_to[i])

            if nearby_points_from.size(0) == 0:
                # print(f"No valid points found for emb_from[{i}], setting emb_from_1 as original emb_from[{i}].")
                emb_from_1_list.append(emb_from[i])
            else:
                # 计算与 ground_truth_from[i] 的相似度
                similarities_from = sim_fn(nearby_semantics_from, ground_truth_from[i].unsqueeze(0).expand_as(nearby_semantics_from), dim=1)

                if similarities_from.numel() > 0:
                    best_from_idx = torch.argmax(similarities_from)
                    emb_from_1 = nearby_points_from[best_from_idx]
                    emb_from_1_list.append(emb_from_1)
                else:
                    # print(f"No valid similarity values for emb_from[{i}], setting emb_from_1 as original emb_from[{i}].")
                    emb_from_1_list.append(emb_from[i])

        # 将最相似点的列表转换为张量
        emb_to_1 = torch.stack(emb_to_1_list)
        emb_from_1 = torch.stack(emb_from_1_list)

        return emb_to_1, emb_from_1

    def filter_neg(self, neg_pred_from, neg_pred_to, delta=1e-1):
        neg_pred_from = neg_pred_from.cpu().detach().numpy()
        neg_pred_to = neg_pred_to.cpu().detach().numpy()
        neg_conf_from =  np.amax(softmax(neg_pred_from, axis=1), axis=1)
        neg_conf_to =  np.amax(softmax(neg_pred_to, axis=1), axis=1)
        neg_pred_edge_from_Res = neg_pred_from.argmax(axis=1)
        neg_pred_edge_to_Res = neg_pred_to.argmax(axis=1)
        condition1 = (neg_pred_edge_from_Res==neg_pred_edge_to_Res)
        condition2 = (neg_conf_from==neg_conf_to)
        # condition2 = (np.abs(neg_conf_from - neg_conf_to)< delta)
        indices = np.where(~(condition1 & condition2))[0]
        return indices
    
    def newton_step_with_regularization(self, edge_to_idx, edge_from_idx, dynamic_margin, is_pred_same, edge_to, edge_from, probs, emb_to, emb_from, curr_model, pred_to_softmax, pred_from_softmax, positive_distance_mean, negative_distance_mean, epsilon=1e-4):
        # Ensure the input tensors require gradient
        for tensor in [edge_to, edge_from, emb_to, emb_from]:
            tensor.requires_grad_(True)

        # umap loss
    
        distance_embedding = torch.norm(emb_to - emb_from, dim=1)
        probabilities_distance = convert_distance_to_probability(distance_embedding, self.a, self.b)
        probabilities_graph = probs
        _, _, ce_loss = compute_cross_entropy(probabilities_graph, probabilities_distance, repulsion_strength=self._repulsion_strength)

        # Create a tensor of ones with the same size as ce_loss
        ones = torch.ones_like(ce_loss)

        # Compute gradient 
        grad = torch.autograd.grad(ce_loss, emb_to, grad_outputs=ones, create_graph=True)[0]
        # Compute gradient for emb_from
        grad_emb_from = torch.autograd.grad(ce_loss, emb_from, grad_outputs=ones, create_graph=True)[0] 

        ################################################################################## analysis grad end  ############################################################################################

        # use learning rate approximate the y_next
        next_emb_to = emb_to - 1 * grad
        next_emb_from = emb_from - 1 * grad_emb_from

        """
        strategy 1: gen y* from yi, and then push y* to yj then calculate || y*-yi || 
        strategy 2: gen y* from yj, and then pull y* to yi then calculate || y*-yi || 
        strategy 3: gen yi* and yj * from yj an yi, and push yi* to yj di = || yi* - yi||,and push yi* to yi, dj = || yj* - yj||, margin = max(di,dj)
        """
        """ strategy 1 """
        # metrix = torch.tensor(next_emb_to, dtype=torch.float, requires_grad=True)
        # strategy 2
        # metrix = torch.tensor(next_emb_from, dtype=torch.float, requires_grad=True)
        """ strategy 3 """
        metrix = torch.tensor(torch.cat((next_emb_to, next_emb_from),dim=0), dtype=torch.float, requires_grad=True)

        for param in curr_model.parameters():
            param.requires_grad = False


        # loss = pred_from_softmax - torch.mean(torch.pow(pred_from_softmax - F.softmax(inv / 0.01, dim=-1), 2),1)
        optimizer = optim.Adam([metrix], lr=0.01)
        # 训练循环
        for epoch in range(20):
            optimizer.zero_grad() 
            inv = curr_model.decoder(metrix) 
            inv_pred = self.pred_fn(inv)
            # inv_pred = torch.tensor(inv_pred, dtype=torch.float, device=self.DEVICE, requires_grad=True)
            # # 计算损失
            
            """ strategy 1 """
            # inv_pred_to_softmax = F.softmax(inv_pred / 0.001, dim=-1)
            # loss = 10 * torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2)) + torch.mean(torch.pow(inv - edge_from, 2))
            # strategy 2   
            
                # Calculate the three terms separately
                # first_term = torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2))
                # third_term = torch.mean(torch.pow(emb_to - metrix, 2))
                # threshold = 0.01  # 

                # if first_term.item() < threshold:
                #     loss = first_term + torch.mean(torch.pow(inv - edge_from, 2)) + 0.1 * third_term
                # else:
                #     loss = first_term + torch.mean(torch.pow(inv - edge_from, 2))
                                             # 
                # loss = 100 * torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2)) + torch.mean(torch.pow(inv - edge_from, 2)) + 0.1 * torch.mean(torch.pow(emb_to - metrix, 2))
            """ strategy 3 """
            inv_pred_softmax = F.softmax(inv_pred / 0.001, dim=-1)
            loss = 10 * torch.mean(torch.pow(torch.cat((pred_from_softmax,pred_to_softmax),dim=0) - inv_pred_softmax, 2)) + torch.mean(torch.pow(inv - torch.cat((edge_from, edge_to),dim=0), 2))

            # bp
            loss.backward(retain_graph=True)         
            optimizer.step()

            # if loss.item() < 0.1:
            #     print(f"Stopping early at epoch {epoch} as loss dropped below 0.1")
            #     break
            # if epoch % 500 == 0:
                # print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        """strategy 1 or 2"""

        # final_margin = torch.norm(emb_to - metrix, dim=1)

        """strategy 3 start"""
        margin = torch.norm( torch.cat((emb_to, emb_from),dim=0) - metrix, dim=1)
        total_length = margin.size(0)
        half_length = total_length // 2
        margin_to = margin[:half_length]
        margin_from = margin[half_length:]    
        final_margin =  torch.max(margin_to, margin_from)
        """strategy 3 end """

        # final_margin = torch.where(final_margin < positive_distance_mean.item(), dynamic_margin[~is_pred_same], final_margin)
        
        
        # margin = torch.where(margin > negative_distance_mean.item(), dynamic_margin[~is_pred_same], margin)

        final_margin = torch.max(final_margin, dynamic_margin[~is_pred_same])

        for param in curr_model.parameters():
            param.requires_grad = True

        dynamic_margin[~is_pred_same] = final_margin.to(self.DEVICE)



        return dynamic_margin

    def newton_step_with_regularization_new(self, edge_to_idx, edge_from_idx, dynamic_margin, is_conf_same, edge_to, edge_from, probs, emb_to, emb_from, curr_model, pred_to_softmax, pred_from_softmax, positive_distance_mean, negative_distance_mean, epsilon=1e-4):
        # Ensure the input tensors require gradient
        for tensor in [edge_to, edge_from, emb_to, emb_from]:
            tensor.requires_grad_(True)

        # umap loss
    
        distance_embedding = torch.norm(emb_to - emb_from, dim=1)
        probabilities_distance = convert_distance_to_probability(distance_embedding, self.a, self.b)
        probabilities_graph = probs
        _, _, ce_loss = compute_cross_entropy(probabilities_graph, probabilities_distance, repulsion_strength=self._repulsion_strength)

        # Create a tensor of ones with the same size as ce_loss
        ones = torch.ones_like(ce_loss)

        # Compute gradient 
        grad = torch.autograd.grad(ce_loss, emb_to, grad_outputs=ones, create_graph=True)[0]
        # Compute gradient for emb_from
        grad_emb_from = torch.autograd.grad(ce_loss, emb_from, grad_outputs=ones, create_graph=True)[0] 

        ################################################################################## analysis grad end  ############################################################################################

        # use learning rate approximate the y_next
        next_emb_to = emb_to - 1 * grad
        next_emb_from = emb_from - 1 * grad_emb_from

        """
        strategy 1: gen y* from yi, and then push y* to yj then calculate || y*-yi || 
        strategy 2: gen y* from yj, and then pull y* to yi then calculate || y*-yi || 
        strategy 3: gen yi* and yj * from yj an yi, and push yi* to yj di = || yi* - yi||,and push yi* to yi, dj = || yj* - yj||, margin = max(di,dj)
        """
        """ strategy 1 """
        # metrix = torch.tensor(next_emb_to, dtype=torch.float, requires_grad=True)
        # strategy 2
        # metrix = torch.tensor(next_emb_from, dtype=torch.float, requires_grad=True)
        """ strategy 3 """
        metrix = torch.tensor(torch.cat((next_emb_to, next_emb_from),dim=0), dtype=torch.float, requires_grad=True)

        for param in curr_model.parameters():
            param.requires_grad = False


        # loss = pred_from_softmax - torch.mean(torch.pow(pred_from_softmax - F.softmax(inv / 0.01, dim=-1), 2),1)
        optimizer = optim.Adam([metrix], lr=0.01)
        # 训练循环
        for epoch in range(20):
            optimizer.zero_grad() 
            inv = curr_model.decoder(metrix) 
            inv_pred = self.pred_fn(inv)
            # inv_pred = torch.tensor(inv_pred, dtype=torch.float, device=self.DEVICE, requires_grad=True)
            # # 计算损失
            
            """ strategy 1 """
            # inv_pred_to_softmax = F.softmax(inv_pred / 0.001, dim=-1)
            # loss = 10 * torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2)) + torch.mean(torch.pow(inv - edge_from, 2))
            # strategy 2   
            
                # Calculate the three terms separately
                # first_term = torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2))
                # third_term = torch.mean(torch.pow(emb_to - metrix, 2))
                # threshold = 0.01  # 

                # if first_term.item() < threshold:
                #     loss = first_term + torch.mean(torch.pow(inv - edge_from, 2)) + 0.1 * third_term
                # else:
                #     loss = first_term + torch.mean(torch.pow(inv - edge_from, 2))
                                             # 
                # loss = 100 * torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2)) + torch.mean(torch.pow(inv - edge_from, 2)) + 0.1 * torch.mean(torch.pow(emb_to - metrix, 2))
            """ strategy 3 """
            inv_pred_softmax = F.softmax(inv_pred / 0.001, dim=-1)
            loss = 10 * torch.mean(torch.pow(torch.cat((pred_to_softmax,pred_from_softmax),dim=0) - inv_pred_softmax, 2)) + torch.mean(torch.pow(inv - torch.cat((edge_to, edge_from),dim=0), 2))

            # bp
            loss.backward(retain_graph=True)         
            optimizer.step()
        
        """strategy 1 or 2"""

        # final_margin = torch.norm(emb_to - metrix, dim=1)

        """strategy 3 start"""
        margin = torch.norm(torch.cat((emb_to, emb_from),dim=0) - metrix, dim=1)
        total_length = margin.size(0)
        half_length = total_length // 2
        margin_to = margin[:half_length]
        margin_from = margin[half_length:]    
        final_margin =  torch.max(margin_to, margin_from)
        """strategy 3 end """

        # final_margin = torch.where(final_margin < positive_distance_mean.item(), dynamic_margin[~is_conf_same], final_margin)
        
        
        # margin = torch.where(margin > negative_distance_mean.item(), dynamic_margin[~is_conf_same], margin)

        final_margin = torch.max(final_margin, dynamic_margin[~is_conf_same])

        for param in curr_model.parameters():
            param.requires_grad = True

        dynamic_margin[~is_conf_same] = final_margin.to(self.DEVICE)

        return dynamic_margin
    
class UmapLoss_refine_conf(nn.Module):
    def __init__(self, negative_sample_rate, device,  data_provider, epoch, net, error_conf, neg_grid, pos_grid,distance_list,fixed_number = 5, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(UmapLoss_refine_conf, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)
        self.data_provider = data_provider
        self.epoch = epoch
        self.net = net
        self.model_path = os.path.join(self.data_provider.content_path, "Model")
        self.fixed_number = fixed_number
        
        self.error_conf = torch.tensor(error_conf)
        self.neg_grid = torch.tensor(neg_grid)
        self.pos_grid = torch.tensor(pos_grid)
        self.distance_list = distance_list

        model_location = os.path.join(self.model_path, "{}_{:d}".format('Epoch', epoch), "subject_model.pth")
        self.net.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")),strict=False)
        self.net.to(self.DEVICE)
        self.net.train()

        for param in net.parameters():
            param.requires_grad = False

        self.pred_fn = self.net.prediction

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, edge_to_idx, edge_from_idx, embedding_to, embedding_from, probs, pred_edge_to, pred_edge_from,edge_to, edge_from,recon_to, recon_from,a_to, a_from,recon_pred_edge_to,recon_pred_edge_from,curr_model,iteration):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        pred_edge_to_neg_Res = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        pred_repeat_neg = torch.repeat_interleave(pred_edge_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]
        pred_edge_from_neg_Res = pred_repeat_neg[randperm]
        indicates = self.filter_neg(pred_edge_from_neg_Res, pred_edge_to_neg_Res)

        #### strategy confidence: filter negative
        embedding_neg_to = embedding_neg_to[indicates]
        embedding_neg_from = embedding_neg_from[indicates]
        

        neg_num = len(embedding_neg_from)
        
        #### identify if the conf error is in the current pair #####################################
        conf_e_neg_from = []
        conf_e_pos_from = []
        conf_e_to_pos = []
        conf_e_to_neg = []

        ####### label the element that in the error conf
        mask_to = torch.isin(edge_to_idx, self.error_conf)
        mask_from = torch.isin(edge_from_idx, self.error_conf)
        ######## get the indicates
        conf_e_indices_to = torch.nonzero(mask_to, as_tuple=True)[0]
        conf_e_indices_from = torch.nonzero(mask_from, as_tuple=True)[0]

        filtered_to_idx = edge_to_idx[conf_e_indices_to]
        filtered_from_idx = edge_from_idx[conf_e_indices_from]

        for i,org_index in enumerate(filtered_to_idx):
            ##### get the position in the error_conf
            indicates = (self.error_conf == org_index).nonzero(as_tuple=True)[0]
            indicates= indicates[0].item()
            neg_grids = self.neg_grid[indicates].squeeze()
            # pos_grids = self.pos_grid[indicates].squeeze()
            pos_grids = self.pos_grid[indicates].unsqueeze(0) if self.pos_grid[indicates].ndim == 1 else self.pos_grid[indicates]
            
            # Append the matched negative grids and the corresponding embedding
            # org_emb = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
            conf_e_neg_from.extend([embedding_to[i]] * len(neg_grids)) 
            conf_e_to_neg.extend(neg_grids)
            
            if self.distance_list[indicates] < 0.5:
                # print("indicates to",indicates)
                conf_e_pos_from.extend([embedding_from[i]] * len(pos_grids)) 
                conf_e_to_pos.extend(pos_grids)
        
        for i,org_index in enumerate(filtered_from_idx):
            indicates = (self.error_conf == org_index).nonzero(as_tuple=True)[0]
            indicates= indicates[0].item()
            neg_grids = self.neg_grid[indicates].squeeze()
            # pos_grids = self.pos_grid[indicates].squeeze()
            pos_grids = self.pos_grid[indicates].unsqueeze(0) if self.pos_grid[indicates].ndim == 1 else self.pos_grid[indicates]
            
            # Append the matched negative grids and the corresponding embedding
            # org_emb = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
            conf_e_neg_from.extend([embedding_from[i]] * len(neg_grids)) 
            conf_e_to_neg.extend(neg_grids)
            if self.distance_list[indicates] < 1:
                # print("indicates from",indicates)
                conf_e_pos_from.extend([embedding_from[i]] * len(pos_grids)) 
                conf_e_to_pos.extend(pos_grids)
        
        # Convert lists to tensors
        if len(conf_e_to_pos) > 0:
            conf_e_neg_from = torch.stack(conf_e_neg_from).to(self.DEVICE)
            conf_e_pos_from = torch.stack(conf_e_pos_from).to(self.DEVICE)
            conf_e_to_pos = torch.stack(conf_e_to_pos).to(self.DEVICE)
            conf_e_to_neg = torch.stack(conf_e_to_neg).to(self.DEVICE)
            # print("conf_e_pos_from",conf_e_pos_from.shape,"conf_e_to_pos",conf_e_to_pos.shape,"conf_e_from - conf_e_to_neg",conf_e_from.shape,conf_e_to_neg.shape)
            conf_pos_distance =  torch.norm(conf_e_pos_from - conf_e_to_pos, dim=1).to(self.DEVICE)
            conf_neg_distance = torch.norm(conf_e_neg_from - conf_e_to_neg, dim=1).to(self.DEVICE)
        else:
            conf_pos_distance = torch.tensor([], device=self.DEVICE)
            conf_neg_distance = torch.tensor([], device=self.DEVICE)
            
 

        positive_distance = torch.norm(embedding_to - embedding_from, dim=1)
        negative_distance = torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
        #  distances between samples (and negative samples)
        positive_distance_mean = torch.mean(positive_distance)
        negative_distance_mean = torch.mean(negative_distance)

        #### dynamic labeling
        pred_edge_to_Res = pred_edge_to.argmax(axis=1)
        pred_edge_from_Res = pred_edge_from.argmax(axis=1)

        is_pred_same = (pred_edge_to_Res.to(self.DEVICE) == pred_edge_from_Res.to(self.DEVICE))
        is_pred_same = is_pred_same.to(self.DEVICE)
        pred_edge_to = pred_edge_to.to(self.DEVICE)
        pred_edge_from = pred_edge_from.to(self.DEVICE)

        recon_pred_to_Res = recon_pred_edge_to.argmax(axis=1)
        recon_pred_from_Res = recon_pred_edge_from.argmax(axis=1)


        temp = 0.001
        recon_pred_to_softmax = F.softmax(recon_pred_edge_to / temp, dim=-1)
        recon_pred_from_softmax = F.softmax(recon_pred_edge_from / temp, dim=-1)

        pred_to_softmax = F.softmax(pred_edge_to / temp, dim=-1)
        pred_from_softmax = F.softmax(pred_edge_from / temp, dim=-1)
        
        recon_pred_to_softmax = torch.Tensor(recon_pred_to_softmax.to(self.DEVICE))
        recon_pred_from_softmax = torch.Tensor(recon_pred_from_softmax.to(self.DEVICE))
        #### umap loss
        distance_embedding = torch.cat(
            (
                positive_distance,
                conf_pos_distance,
                negative_distance,
                conf_neg_distance
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        probabilities_graph = torch.cat(
        (
            probs,
            torch.ones(len(conf_pos_distance)).to(self.DEVICE), # ground truth grid points
            # torch.zeros(len(conf_neg_distance)).to(self.DEVICE)
            torch.zeros(neg_num + len(conf_neg_distance)).to(self.DEVICE)
            # torch.zeros(neg_num + len(conf_neg_distance)).to(self.DEVICE)
        ),dim=0)

        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )  


        batch_margin = positive_distance_mean +  (negative_distance_mean - positive_distance_mean) * (1-probs)
        init_margin = (1.0 - is_pred_same.float()) * batch_margin

        margin = init_margin
        
        margin_loss = F.relu(margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()
        
        umap_l = torch.mean(ce_loss).to(self.DEVICE) 
        if torch.isnan(umap_l):
            umap_l = torch.tensor(0.0).to(self.DEVICE)
        margin_loss = margin_loss.to(self.DEVICE)

        if torch.isnan(margin_loss):
            margin_loss = torch.tensor(0.0).to(margin_loss.device)

        return umap_l, margin_loss, umap_l+margin_loss

    def filter_neg(self, neg_pred_from, neg_pred_to, delta=1e-1):
        neg_pred_from = neg_pred_from.cpu().detach().numpy()
        neg_pred_to = neg_pred_to.cpu().detach().numpy()
        neg_conf_from =  np.amax(softmax(neg_pred_from, axis=1), axis=1)
        neg_conf_to =  np.amax(softmax(neg_pred_to, axis=1), axis=1)
        neg_pred_edge_from_Res = neg_pred_from.argmax(axis=1)
        neg_pred_edge_to_Res = neg_pred_to.argmax(axis=1)
        condition1 = (neg_pred_edge_from_Res==neg_pred_edge_to_Res)
        condition2 = (neg_conf_from==neg_conf_to)
        # condition2 = (np.abs(neg_conf_from - neg_conf_to)< delta)
        indices = np.where(~(condition1 & condition2))[0]
        return indices   

class DVILoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, lambd1, lambd2, device):
        super(DVILoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.device = device

    def forward(self, edge_to_idx, edge_from_idx, edge_to, edge_from, a_to, a_from, curr_model,probs,pred_edge_to, pred_edge_from,recon_pred_edge_to,recon_pred_edge_from,iteration,data_provider,epoch,points_2d):
      
        outputs = curr_model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]


        # TODO stop gradient edge_to_ng = edge_to.detach().clone()

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l,new_l,total_l = self.umap_loss(edge_to_idx, edge_from_idx, embedding_to, embedding_from, probs,pred_edge_to, pred_edge_from,edge_to, edge_from,recon_to, recon_from,a_to, a_from,recon_pred_edge_to,recon_pred_edge_from, curr_model,iteration,data_provider,epoch, points_2d)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        loss = total_l + self.lambd1 * recon_l + self.lambd2 * temporal_l

        return umap_l, new_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss
    


class ReconstructionLoss(nn.Module):
    def __init__(self, beta=1.0,alpha=0.5,scale_factor=0.1):
        super(ReconstructionLoss, self).__init__()
        self._beta = beta
        self._alpha = alpha
        self.scale_factor = scale_factor

    def forward(self, edge_to, edge_from, recon_to, recon_from, a_to, a_from):
        loss1 = torch.mean(torch.mean(torch.multiply(torch.pow((1+a_to), self._beta), torch.pow(edge_to - recon_to, 2)), 1))
        loss2 = torch.mean(torch.mean(torch.multiply(torch.pow((1+a_from), self._beta), torch.pow(edge_from - recon_from, 2)), 1))
        return (loss1 + loss2) /2


# TODO delete
class BoundaryAwareLoss(nn.Module):
    def __init__(self, umap_loss, device, scale_factor=0.1,margin=3):
        super(BoundaryAwareLoss, self).__init__()
        self.umap_loss = umap_loss
        self.device = device
        self.scale_factor = scale_factor
        self.margin = margin
    
    def forward(self, edge_to, edge_from, model,probs):
        outputs = model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]     
        recon_to, recon_from = outputs["recon"]
        
        # reconstruction loss - recon_to, recon_from close to edge_to, edge_from
        reconstruction_loss_to = F.mse_loss(recon_to, edge_to)
        reconstruction_loss_from = F.mse_loss(recon_from, edge_from)
        recon_loss = reconstruction_loss_to + reconstruction_loss_from


        umap_l = self.umap_loss(embedding_to, embedding_from, edge_to, edge_from, recon_to, recon_from, probs, self.margin).to(self.device)
 
        # return self.scale_factor * umap_l +  0.2 * recon_loss
        return 0.1 * umap_l + 0.1* recon_loss
class BoundaryDistanceConsistencyLoss(nn.Module):
    def __init__(self, data_provider, iteration, device):
        super(BoundaryDistanceConsistencyLoss, self).__init__()
        self.data_provider = data_provider
        self.iteration = iteration
        self.device = device
    def forward(self, samples, recon_samples):
        combined_samples = torch.cat((samples, recon_samples), dim=0)
        combined_probs = self.data_provider.get_pred(self.iteration, combined_samples.cpu().detach().numpy(),0)
   
        original_probs, recon_probs = np.split(combined_probs, 2, axis=0)

        original_boundary_distances = self.calculate_boundary_distances(original_probs)
        recon_boundary_distances = self.calculate_boundary_distances(recon_probs)
        
        correlation, _ = spearmanr(original_boundary_distances, recon_boundary_distances)
        consistency_loss = 1 - abs(correlation)

        return torch.tensor(consistency_loss, requires_grad=True)

    def calculate_boundary_distances(self, probs):
        top_two_probs = np.sort(probs, axis=1)[:, -2:]
        return top_two_probs[:, 1] - top_two_probs[:, 0]

        
class TrustvisLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, bon_con_loss, lambd1, lambd2, device):
        super(TrustvisLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.bon_con_loss = bon_con_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.device = device

    def forward(self, edge_to, edge_from, a_to, a_from, curr_model):
        outputs = curr_model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        # TODO stop gradient edge_to_ng = edge_to.detach().clone()

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from).to(self.device)
        temporal_l = self.temporal_loss(curr_model).to(self.device)
        bon_con_loss = self.bon_con_loss(torch.cat((edge_to,edge_from), dim=0), torch.cat((recon_to,recon_from), dim=0))

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l + bon_con_loss

        return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, bon_con_loss, loss


class SmoothnessLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(SmoothnessLoss, self).__init__()
        self._margin = margin

    def forward(self, embedding, target, Coefficient):
        loss = torch.mean(Coefficient * torch.clamp(torch.norm(embedding-target, dim=1)-self._margin, min=0))
        return loss


class SingleVisLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, lambd):
        super(SingleVisLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.lambd = lambd

    def forward(self, edge_to, edge_from, a_to, a_from, outputs, probs):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from)
        # recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from)
        umap_l = self.umap_loss(embedding_to, embedding_from, probs)

        loss = umap_l + self.lambd * recon_l

        return umap_l, recon_l, loss

class HybridLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, smooth_loss, lambd1, lambd2):
        super(HybridLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.smooth_loss = smooth_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2

    def forward(self, edge_to, edge_from, a_to, a_from, embeded_to, coeff, outputs):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from)
        umap_l = self.umap_loss(embedding_to, embedding_from)
        smooth_l = self.smooth_loss(embedding_to, embeded_to, coeff)

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * smooth_l

        return umap_l, recon_l, smooth_l, loss


class TemporalLoss(nn.Module):
    def __init__(self, prev_w, device) -> None:
        super(TemporalLoss, self).__init__()
        self.prev_w = prev_w
        self.device = device
        for param_name in self.prev_w.keys():
            self.prev_w[param_name] = self.prev_w[param_name].to(device=self.device, dtype=torch.float32)

    def forward(self, curr_module):
        loss = torch.tensor(0., requires_grad=True).to(self.device)
        # c = 0
        for name, curr_param in curr_module.named_parameters():
            # c = c + 1
            prev_param = self.prev_w[name]
            # tf dvi: diff = tf.reduce_sum(tf.math.square(w_current[j] - w_prev[j]))
            loss = loss + torch.sum(torch.square(curr_param-prev_param))
            # loss = loss + torch.norm(curr_param-prev_param, 2)
        # in dvi paper, they dont have this normalization (optional)
        # loss = loss/c
        return loss


class DummyTemporalLoss(nn.Module):
    def __init__(self, device) -> None:
        super(DummyTemporalLoss, self).__init__()
        self.device = device

    def forward(self, curr_module):
        loss = torch.tensor(0., requires_grad=True).to(self.device)
        return loss
    

class PositionRecoverLoss(nn.Module):
    def __init__(self, device) -> None:
        super(PositionRecoverLoss, self).__init__()
        self.device = device
    def forward(self, position, recover_position):
        mse_loss = nn.MSELoss().to(self.device)
        loss = mse_loss(position, recover_position)
        return loss



class TrustALLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, lambd1, lambd2, device):
        super(TrustALLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.device = device

    def forward(self, edge_to, edge_from, a_to, a_from, curr_model, outputs, edge_to_pred, edge_from_pred):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        # TODO stop gradient edge_to_ng = edge_to.detach().clone()

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from,edge_to_pred, edge_from_pred).to(self.device)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l

        return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss
        
       

class DVIALLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, lambd1, lambd2, lambd3, device):
        super(DVIALLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.lambd3 = lambd3
        self.device = device

        # self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()


    def forward(self, edge_to, edge_from, a_to, a_from, curr_model, outputs,data):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        # TODO stop gradient edge_to_ng = edge_to.detach().clone()
        # if self.lambd3 != 0:
        #     data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        #     recon_data = curr_model(data,data)['recon'][0]
        #     pred_loss = self.mse_loss(data, recon_data)
        # else:
        #     pred_loss = torch.Tensor(0)
            
        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from).to(self.device)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        if self.lambd3 != 0:
            data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
            recon_data = curr_model(data,data)['recon'][0]
            pred_loss = self.mse_loss(data, recon_data)
            loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l + self.lambd3 * pred_loss
            return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss, pred_loss
        else:
            loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l 
            pred_loss = torch.tensor(0.0).to(self.device)

        return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss, pred_loss 



class ActiveLearningLoss(nn.Module):
    def __init__(self, data_provider, iteration, device):
        super(ActiveLearningLoss, self).__init__()
        self.data_provider = data_provider
        self.iteration = iteration
        self.device = device

    def forward(self, curr_model,data):
         
        self.data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        recon_data = curr_model(self.data,self.data)['recon'][0]
        loss = self.cross_entropy_loss(self.data, recon_data)
        # normalized_loss = torch.sigmoid(loss)
        return  loss


# def find_same_confidence(pred_edge_to_prob, pred_edge_from_prob, interval=0.2):
#     # 判断置信度区间
#     edge_to_interval = torch.floor(pred_edge_to_prob / interval).int()
#     edge_from_interval = torch.floor(pred_edge_from_prob / interval).int()

#     # 判断是否落在同一置信度区间
#     is_conf_in_same_interval = edge_to_interval == edge_from_interval

#     # np.save('/home/yiming/cophi/projects/Trustvis/debugging_observation/is_conf_in_same_interval.npy', is_conf_in_same_interval.cpu().numpy())

#     # 返回形状为 (N,) 的张量，满足条件的是 1，不满足的是 0
#     return is_conf_in_same_interval

def find_same_confidence(pred_edge_to_prob, pred_edge_from_prob, threshold=0.2):
    """
    判断置信度之间的差别是否小于某个阈值。

    Parameters:
    pred_edge_to_prob (torch.Tensor): 第一个置信度张量。
    pred_edge_from_prob (torch.Tensor): 第二个置信度张量。
    threshold (float): 允许的置信度差异阈值。

    Returns:
    torch.Tensor: 形状为 (N,) 的布尔张量，表示置信度差别是否小于阈值。
    """

    # 计算置信度之间的绝对差值
    confidence_diff = torch.abs(pred_edge_to_prob - pred_edge_from_prob)

    # 判断差别是否小于阈值
    is_conf_in_same_range = confidence_diff < threshold

    # 返回布尔张量，表示差别小于阈值的元素
    return is_conf_in_same_range

def find_same_confidence_recon(pred_edge_to_prob, pred_edge_from_prob, 
                                        recon_pred_edge_to_prob, recon_pred_edge_from_prob, 
                                        interval=0.2):
    # 计算区间（置信度除以 interval）
    edge_to_interval = torch.floor(pred_edge_to_prob / interval).int()
    edge_from_interval = torch.floor(pred_edge_from_prob / interval).int()
    
    recon_edge_to_interval = torch.floor(recon_pred_edge_to_prob / interval).int()
    recon_edge_from_interval = torch.floor(recon_pred_edge_from_prob / interval).int()

    # 判断投影前后 confidence 是否在相同区间内
    is_conf_in_same_interval_to = edge_to_interval == recon_edge_to_interval
    is_conf_in_same_interval_from = edge_from_interval == recon_edge_from_interval

    # 找出同时满足投影前后 confidence 一致的样本
    is_recon_conf_in_same_interval = is_conf_in_same_interval_to & is_conf_in_same_interval_from

    return is_recon_conf_in_same_interval

def compute_pred_confidence_diff(predictions):
    """
    计算每个样本分类结果的置信度差异，衡量模型对分类结果的确定性。

    Parameters:
    predictions (np.ndarray): 模型的预测结果，形状为 (num_samples, num_classes)，
                              其中每一行代表一个样本在不同类别上的预测概率。

    Returns:
    np.ndarray: 每个样本的置信度差异值，形状为 (num_samples,)。
    """
    # 避免除以零的数值不稳定问题
    predictions = predictions + 1e-8

    # 对预测结果进行排序
    sorted_preds, _ = torch.sort(predictions, dim=1)

    # 计算最大置信度和次大置信度的差异，并归一化
    pred_confidence_diff = (sorted_preds[:, -1] - sorted_preds[:, -2]) / (sorted_preds[:, -1] - sorted_preds[:, 0])

    return pred_confidence_diff

def compute_recon_pred_confidence_diff(predictions, indices, indices_second):
    """
    计算每个样本分类结果的置信度差异，并确保当前预测结果不变。
    对于不满足 max_conf_is_correct 的样本，提供额外的推力。

    Parameters:
    predictions (torch.Tensor): 模型的预测结果，形状为 (num_samples, num_classes)，
                                每一行代表一个样本在不同类别上的预测概率。
    indices (torch.Tensor): 每行的最大置信度所在的列索引，代表原始预测结果。

    Returns:
    torch.Tensor: 每个样本的置信度差异值，形状为 (num_samples,)。
    """
    # 避免除以零的数值不稳定问题
    predictions = predictions + 1e-8

    # 对预测结果进行排序
    sorted_preds, sorted_indices = torch.sort(predictions, dim=1)

    # 计算每个样本当前的最大置信度是否来自原始预测结果（根据 indices）
    max_conf_is_correct = sorted_indices[:, -1] == indices

    # 计算最大置信度和次大置信度的差异
    pred_confidence_diff = torch.zeros(predictions.shape[0], device=predictions.device)

    # 获取这些索引的预测值
    first_pred_vals = predictions[torch.arange(predictions.size(0)), indices]
    second_pred_vals = predictions[torch.arange(predictions.size(0)), indices_second]

    # 对于预测结果没有改变的样本，选择这些样本并计算 diff
    if max_conf_is_correct.any():  # 如果有满足条件的样本
        # print("len(max_conf_is_correct)", torch.sum(max_conf_is_correct))
        correct_sorted_preds = sorted_preds[max_conf_is_correct]

        pred_confidence_diff[max_conf_is_correct] = (
            (first_pred_vals[max_conf_is_correct] - second_pred_vals[max_conf_is_correct]) / 
            (first_pred_vals[max_conf_is_correct] - correct_sorted_preds[:, 0])
        )
    # 对于 max_conf_is_correct 为 False 的样本，提供额外推力
    pred_confidence_diff[~max_conf_is_correct] = 0.0 

    # pred_confidence_diff = (
    #     (first_pred_vals - second_pred_vals) / 
    #     (first_pred_vals - sorted_preds[:, 0])
    # )

    return pred_confidence_diff


# class ContourLoss(nn.Module):
#     def __init__(self, all_nodes, all_nodes_2d_dict, all_nodes_2d, device):
#         super(ContourLoss, self).__init__()
#         self.all_nodes = all_nodes
#         self.all_nodes_2d_dict = all_nodes_2d_dict
#         self.all_nodes_2d = all_nodes_2d
#         self.device = device
    
#     def forward(self, logits, train_data, model):
#         """
#         :param train_data: 高维训练数据 (batch_size, 512)
#         :param model: 用于降维的模型
#         :return: 平均投影损失
#         """
#         # 1. 将高维训练数据传入模型，得到二维的投影点
#         # emb_data = model(train_data, train_data)['umap'][0]
#         emb_data = model(train_data)

#         # 2. 计算预测语义的 softmax
#         softmax_probs = F.softmax(logits, dim=1)  # softmax 概率 (batch_size, n_classes)

#         # 3. 计算与 all_nodes_array 的余弦相似度
#         cosine_sim = F.cosine_similarity(softmax_probs.unsqueeze(1), self.all_nodes.unsqueeze(0), dim=2)
        
#         # 4. 找到每个数据点最相似的 all_nodes_array
#         closest_node_indices = torch.argmax(cosine_sim, dim=1).cpu().numpy()  # (batch_size,)

#         # 5. 通过字典找到最近的 all_nodes 对应的二维投影区域
#         total_proj_loss = 0.0
#         for i, idx in enumerate(closest_node_indices):
#             if idx in self.all_nodes_2d_dict:
#                 # pixel_points = self.all_nodes_2d_dict[idx]  # 获取二维区域
#                 # center_of_mass = torch.mean(torch.tensor(pixel_points, dtype=torch.float32, device=self.device), dim=0)  # 区域的中心
#                 # # 计算投影损失，点与中心的距离
#                 # proj_loss = F.mse_loss(emb_data[i], center_of_mass).float()
#                 proj_loss = F.mse_loss(emb_data[i], self.all_nodes_2d[idx]).float()
#                 total_proj_loss += proj_loss

#         # 6. 计算平均投影损失
#         avg_proj_loss = total_proj_loss / train_data.shape[0]
        
#         return avg_proj_loss

class ContourLoss(nn.Module):
    def __init__(self, all_nodes, all_nodes_2d_dict, all_nodes_2d, device):
        super(ContourLoss, self).__init__()
        self.all_nodes = all_nodes
        self.all_nodes_2d_dict = all_nodes_2d_dict
        self.all_nodes_2d = all_nodes_2d
        self.device = device
    
    def forward(self, logits, train_data, model):
        """
        :param logits: 训练数据的预测logits (batch_size, n_classes)
        :param train_data: 高维训练数据 (batch_size, 512)
        :param model: 用于降维的模型
        :return: 平均投影损失
        """
        # 1. 将高维训练数据传入模型，得到二维的投影点
        emb_data = model(train_data)

        # 2. 计算预测语义的 softmax
        softmax_probs = F.softmax(logits, dim=1)  # softmax 概率 (batch_size, n_classes)

        # 3. 找到每个数据点的预测类别
        pred_classes = torch.argmax(softmax_probs, dim=1)  # (batch_size,)

        total_proj_loss = 0.0

        # 4. 遍历每个数据点，筛选 prediction 一致的 all_nodes
        for i in range(train_data.shape[0]):
            pred_class = pred_classes[i]  # 当前数据点的预测类别

            # 筛选与当前预测类别一致的 all_nodes
            mask = torch.argmax(self.all_nodes, dim=1) == pred_class
            filtered_all_nodes = self.all_nodes[mask]
            filtered_all_nodes_2d = self.all_nodes_2d[mask]

            # 计算与筛选后的 all_nodes 的余弦相似度
            cosine_sim = F.cosine_similarity(softmax_probs[i].unsqueeze(0), filtered_all_nodes.unsqueeze(0), dim=2)

            # 找到最相似的节点索引
            closest_node_idx = torch.argmax(cosine_sim).item()

            # 通过字典找到最近的 all_nodes 对应的二维投影位置
            # if closest_node_idx in self.all_nodes_2d_dict:
            proj_loss = F.mse_loss(emb_data[i], filtered_all_nodes_2d[closest_node_idx]).float()
            total_proj_loss += proj_loss

        # 5. 计算平均投影损失
        avg_proj_loss = total_proj_loss / train_data.shape[0]
        
        return avg_proj_loss