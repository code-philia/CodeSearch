# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
from torch.nn import CrossEntropyLoss, MSELoss
from losses import NTXentLoss,AlignLoss
import torch.nn.functional as F
import random
import json
import os
from itertools import chain
import math

# Create new transformer layer for CLS token attention
class CLSAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 12
        self.attention_head_size = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.key.bias)
    def transpose_for_scores(self, x):
        # 1. 首先添加 batch 维度
        if len(x.size()) == 1:  # 对于 CLS token: [768] -> [1, 1, 768]
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.size()) == 2:  # 对于 sequence: [320, 768] -> [1, 320, 768]
            x = x.unsqueeze(0)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, cls_token, sequence_tokens, attention_mask=None):
        # Move linear layers to same device as input tensors
        device = cls_token.device
        self.query = self.query.to(device)
        self.key = self.key.to(device)
        
        # Transform CLS token to query
        cls_query = self.transpose_for_scores(self.query(cls_token))
        
        # Transform all tokens (including CLS) to keys
        seq_keys = self.transpose_for_scores(self.key(sequence_tokens))
        
        # Calculate attention scores
        attention_scores = torch.matmul(cls_query, seq_keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask before softmax
        if attention_mask is not None:
            # Add batch dimension if needed
            if len(attention_mask.size()) == 1:
                attention_mask = attention_mask.unsqueeze(0)
                
            # Expand attention mask to match attention scores dimensions
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(attention_scores.size())
            
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
            
        # Get attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Use original sequence embeddings as values
        sequence_values = self.transpose_for_scores(sequence_tokens)
        
        # Calculate new CLS representation
        new_cls = torch.matmul(attention_probs, sequence_values)
        new_cls = new_cls.permute(0, 2, 1, 3).contiguous()
        new_cls_shape = new_cls.size()[:-2] + (self.all_head_size,)
        new_cls = new_cls.view(new_cls_shape)
        
        return new_cls, attention_probs, attention_scores


class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.cls_attention = CLSAttentionLayer()
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None] 
            # print(self.encoder)
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, output_attentions=True, output_hidden_states=True)
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1), output_attentions=True, output_hidden_states=True)
            # , output_hidden_states=True
        
    def ori_loss(self, code_inputs, code_outputs, nl_outputs):
        #get code and nl vectors
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]
        
        #calculate scores and loss
        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
        
        return loss
        
    def alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, labelled_mapped_tokens, common_mapped_tokens, args, return_vec=False, return_scores=False):
        bs = code_inputs.shape[0]
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]
        # 获取注意力权重
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        for layer_idx in range(len(code_attentions)-1, -1, -1):
            # attention 的形状: (batch_size, num_heads, seq_length, seq_length)
            code_attention = code_attentions[layer_idx].mean(dim=1)  # (batch_size, seq_length, seq_length)
            nl_attention = nl_attentions[layer_idx].mean(dim=1)
            
            # 计算每个 token 对 CLS token 的贡献
            if layer_idx == len(code_attentions)-1:
                code_cls_attention = code_attention[:, 0, :]  # (batch_size, seq_length)
                nl_cls_attention = nl_attention[:, 0, :]
            else:
                code_cls_attention = torch.matmul(code_cls_attention.unsqueeze(1), code_attention).squeeze(1)  # (batch_size, seq_length)
                nl_cls_attention = torch.matmul(nl_cls_attention.unsqueeze(1), nl_attention).squeeze(1)

        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
        
        loss_align_code = 0

        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]
        # print(type(align_outputs_2))
        # aaa

        lcs_pairs = sample_align
        selected_align_outputs_1 = []
        selected_align_outputs_2 = []
        n = 0
        for pair in lcs_pairs:
            # pair[0] 是 comment 的区间，pair[1] 是 code 的区间
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            comment_embeddings = [align_outputs_1[idx+1] for idx in comment_indices]
            # print(len(comment_embeddings))
            comment_mean = torch.mean(torch.stack(comment_embeddings), dim=0)
            # selected_align_outputs_1.append(comment_mean)

            # 计算 code 区间的平均值
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))    
            code_embeddings = [align_outputs_2[idx+1] for idx in code_indices]
            # print(len(code_embeddings))
            code_mean = torch.mean(torch.stack(code_embeddings), dim=0)
            # selected_align_outputs_1.append(code_mean)
            # 计算 code 区间的平均值
            # code_embeddings = []
            # for idx in labelled_mapped_tokens[n]:
            #     if idx < len(align_outputs_2):
            #         code_embeddings.append(align_outputs_2[idx])
            #     else:
            #         break
            # print(len(code_embeddings))
            if code_embeddings:  # 检查 code_embeddings 是否为空
                code_mean = torch.mean(torch.stack(code_embeddings), dim=0)
                selected_align_outputs_1.append(comment_mean)
                selected_align_outputs_2.append(code_mean)
            n += 1
        # 将列表转换为张量
        selected_align_outputs_1 = torch.stack(selected_align_outputs_1)
        selected_align_outputs_2 = torch.stack(selected_align_outputs_2)
        # print(selected_align_outputs_1.shape[0], selected_align_outputs_2.shape[0])

        align_loss = NTXentLoss(args, selected_align_outputs_1.shape[0],temperature=3.0)

        loss_align_code += align_loss(selected_align_outputs_1, selected_align_outputs_2)
        loss_align_code = (loss_align_code/len(lcs_pairs))*2

        # code_cls_attention[local_index], nl_cls_attention[local_index]
        alignment_code_indices = []
        alignment_nl_indices = []

        for n, m in lcs_pairs:
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    alignment_code_indices.extend(range(m[i], m[i + 1] + 1))
            else:
                alignment_code_indices.append(m)
            
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    alignment_nl_indices.extend(range(n[i], n[i + 1] + 1))
            else:
                alignment_nl_indices.append(n)

        # 步骤 1: 展开嵌套列表
        # flat_labelled_mapped_tokens = [item for sublist in labelled_mapped_tokens for item in sublist]
        # 去重处理
        # alignment_code_indices = list(set(flat_labelled_mapped_tokens))
        alignment_code_indices = list(set(alignment_code_indices))
        alignment_nl_indices = list(set(alignment_nl_indices))
        # 给 alignment_code_indices 中的每一个值 +1
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices]
        # print(alignment_code_indices, alignment_nl_indices)

        # 创建 noisy_code_attention 并复制 code_cls_attention 的值
        noisy_code_attention = code_cls_attention[local_index].clone()
        noisy_nl_attention = nl_cls_attention[local_index].clone()
        # print(noisy_code_attention.shape, noisy_nl_attention.shape)

        # # 将 alignment_code_indices 对应位置的值取相反数
        # noisy_code_attention[0, alignment_code_indices] = -code_cls_attention[0, alignment_code_indices]
        # noisy_nl_attention[0, alignment_nl_indices] = -nl_cls_attention[0, alignment_nl_indices]

        # # 计算 noisy token 的 attention loss
        # noisy_attention_loss = self.calculate_noisy_attention_loss(noisy_code_attention, noisy_nl_attention)

        epsilon = 1e-8
        noisy_code_attention = torch.where(noisy_code_attention == 0, torch.full_like(noisy_code_attention, epsilon), noisy_code_attention)
        noisy_nl_attention = torch.where(noisy_nl_attention == 0, torch.full_like(noisy_nl_attention, epsilon), noisy_nl_attention)
        # 初始化四个张量，全为 epsilon
        positive_code_attention = torch.full_like(noisy_code_attention, epsilon)
        negative_code_attention = torch.full_like(noisy_code_attention, epsilon)
        positive_nl_attention = torch.full_like(noisy_nl_attention, epsilon)
        negative_nl_attention = torch.full_like(noisy_nl_attention, epsilon)

        # 创建一个 mask，用于选择正样本位置
        code_mask = torch.zeros_like(noisy_code_attention, dtype=torch.bool)
        nl_mask = torch.zeros_like(noisy_nl_attention, dtype=torch.bool)

        # 对于正样本位置，将值设置为原始的 attention_scores
        code_mask[alignment_code_indices] = 1
        nl_mask[alignment_nl_indices] = 1

        # 更新正样本的得分
        positive_code_attention[code_mask] = noisy_code_attention[code_mask]
        positive_nl_attention[nl_mask] = noisy_nl_attention[nl_mask]

        # 负样本是所有非对齐的位置
        code_neg_mask = ~code_mask
        nl_neg_mask = ~nl_mask

        # 更新负样本的得分
        negative_code_attention[code_neg_mask] = noisy_code_attention[code_neg_mask]
        negative_nl_attention[nl_neg_mask] = noisy_nl_attention[nl_neg_mask]

        # print(positive_code_attention, negative_code_attention)
        # focal loss
        # noisy_attention_loss = (self.focal_attention_loss(positive_code_attention, negative_code_attention) + self.focal_attention_loss(positive_nl_attention, negative_nl_attention))/2

        # regularization loss
        # noisy_attention_loss = (self.regularization_attention_loss(positive_code_attention, negative_code_attention) + self.regularization_attention_loss(positive_nl_attention, negative_nl_attention))/2

        # dynamic_weighting loss
        noisy_attention_loss = (self.dynamic_weighting_attention_loss(positive_code_attention, negative_code_attention) + self.dynamic_weighting_attention_loss(positive_nl_attention, negative_nl_attention))/2
        # print(loss_align_code, noisy_attention_loss)

        if len(common_mapped_tokens) > 0:
            # common_token_loss = self.alignment_loss(align_outputs_2, common_mapped_tokens, args.code_length)
            # 创建一个目标张量，初始值为零
            target = torch.zeros_like(align_outputs_2)
            seq_length = args.code_length
            # 将 common_mapped_tokens 中不在的索引对应的 target 位置设置为 align_outputs_2 中的值
            for idx in range(seq_length):
                if idx not in common_mapped_tokens:
                    target[idx] = align_outputs_2[idx]
            
            # 计算损失
            common_token_loss = F.mse_loss(align_outputs_2, target)
            if noisy_attention_loss > 0:
                total_loss = loss + loss_align_code + noisy_attention_loss + common_token_loss
        else:
            if noisy_attention_loss > 0:
                common_token_loss = 0
                total_loss = loss + loss_align_code + noisy_attention_loss
            else:
                print(noisy_attention_loss)
                print(positive_nl_attention)
                print(negative_nl_attention)
        # total_loss = loss + loss_align_code + noisy_attention_loss

        return loss, total_loss, code_vec, nl_vec, loss_align_code, noisy_attention_loss, common_token_loss

    def attention_alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, labelled_mapped_tokens, common_mapped_tokens, args, return_vec=False, return_scores=False):
        bs = code_inputs.shape[0]
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]
        # 获取注意力权重
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        # all_code_hidden_states = code_outputs.hidden_states

        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[:, :, 0, :].mean(dim=1)
        
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[:, :, 0, :].mean(dim=1)

        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
        
        loss_align_code = 0

        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]

        lcs_pairs = sample_align
        selected_align_outputs_1 = []
        selected_align_outputs_2 = []
        n = 0
        for pair in lcs_pairs:
            # pair[0] 是 comment 的区间，pair[1] 是 code 的区间
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            comment_embeddings = [align_outputs_1[idx+1] for idx in comment_indices]
            comment_mean = torch.mean(torch.stack(comment_embeddings), dim=0)

            # 计算 code 区间的平均值
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))    
            code_embeddings = [align_outputs_2[idx+1] for idx in code_indices]
            code_mean = torch.mean(torch.stack(code_embeddings), dim=0)

            if code_embeddings:  # 检查 code_embeddings 是否为空
                code_mean = torch.mean(torch.stack(code_embeddings), dim=0)
                selected_align_outputs_1.append(comment_mean)
                selected_align_outputs_2.append(code_mean)
            n += 1
        # 将列表转换为张量
        selected_align_outputs_1 = torch.stack(selected_align_outputs_1)
        selected_align_outputs_2 = torch.stack(selected_align_outputs_2)

        align_loss = NTXentLoss(args, selected_align_outputs_1.shape[0],temperature=3.0)

        loss_align_code += align_loss(selected_align_outputs_1, selected_align_outputs_2)
        loss_align_code = (loss_align_code/len(lcs_pairs))*2

        alignment_code_indices = []
        alignment_nl_indices = []

        for n, m in lcs_pairs:
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    alignment_code_indices.extend(range(m[i], m[i + 1] + 1))
            else:
                alignment_code_indices.append(m)
            
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    alignment_nl_indices.extend(range(n[i], n[i + 1] + 1))
            else:
                alignment_nl_indices.append(n)

        # 步骤 1: 展开嵌套列表
        # flat_labelled_mapped_tokens = [item for sublist in labelled_mapped_tokens for item in sublist]
        # 去重处理
        # alignment_code_indices = list(set(flat_labelled_mapped_tokens))
        alignment_code_indices = list(set(alignment_code_indices))
        alignment_nl_indices = list(set(alignment_nl_indices))
        # 给 alignment_code_indices 中的每一个值 +1
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices]
        # print(alignment_code_indices, alignment_nl_indices)

        # Step 1: 计算 attention loss
        certain_code_cls_attention = code_cls_attention[local_index]
        certain_nl_cls_attention = nl_cls_attention[local_index]
        code_attention_loss = self.calculate_attention_loss(certain_code_cls_attention, alignment_code_indices)
        nl_attention_loss = self.calculate_attention_loss(certain_nl_cls_attention, alignment_nl_indices)

        # Step 2: 合并代码和自然语言的 attention loss
        noisy_attention_loss = (code_attention_loss + nl_attention_loss) / 2

        if len(common_mapped_tokens) > 0:
            # common_token_loss = self.alignment_loss(align_outputs_2, common_mapped_tokens, args.code_length)
            # 创建一个目标张量，初始值为零
            target = torch.zeros_like(align_outputs_2)
            seq_length = args.code_length
            # 将 common_mapped_tokens 中不在的索引对应的 target 位置设置为 align_outputs_2 中的值
            for idx in range(seq_length):
                if idx not in common_mapped_tokens:
                    target[idx] = align_outputs_2[idx]
            
            # 计算损失
            common_token_loss = F.mse_loss(align_outputs_2, target)
            if noisy_attention_loss > 0:
                total_loss = loss + loss_align_code + noisy_attention_loss + common_token_loss
        else:
            if noisy_attention_loss > 0:
                common_token_loss = 0
                total_loss = loss + loss_align_code + noisy_attention_loss
            else:
                print(noisy_attention_loss)
                # print(positive_nl_attention)
                # print(negative_nl_attention)
        # total_loss = loss + loss_align_code + noisy_attention_loss

        return loss, total_loss, code_vec, nl_vec, loss_align_code, noisy_attention_loss, common_token_loss

    def calculate_attention_loss(self, attention, alignment_indices):
        """
        计算 attention loss，目标是：
        - 对齐的 token 的 attention 越大越好（接近 1）
        - 非对齐的 token 的 attention 越小越好（接近 0）
        """        
        # 创建目标 attention 分布
        target_attention = torch.zeros_like(attention)
        
        # 对对齐的 token，将目标 attention 设置为 1
        for idx in alignment_indices:  # 遍历对齐的 token 索引
            target_attention[idx] = 1.0  # 期望这些 token 的 attention 是 1
        
        # 使用 MSE loss 来使对齐的 token attention 趋向 1，非对齐的 token attention 趋向 0
        attention_loss = F.mse_loss(attention, target_attention)

        return attention_loss
    
    def alignment_loss(align_outputs_2, common_mapped_tokens, seq_length, alpha=1.0):
        """
        计算对齐损失，使得对于 common_mapped_tokens 中的索引, align_outputs_2 中的隐藏状态尽可能接近零。
        
        Args:
        - align_outputs_2 (torch.Tensor): 模型输出的最后隐藏状态，形状为 (seq_length, hidden_state_dim)。
        - common_mapped_tokens (list of int): 需要对齐的索引。
        - alpha (float): 损失的权重因子。
        
        Returns:
        - loss (torch.Tensor): 计算得到的损失值。
        """ 
        # 创建一个目标张量，初始值为零
        target = torch.zeros_like(align_outputs_2)
        
        # 将 common_mapped_tokens 中不在的索引对应的 target 位置设置为 align_outputs_2 中的值
        for idx in range(seq_length):
            if idx not in common_mapped_tokens:
                target[idx] = align_outputs_2[idx]
        
        # 计算损失
        loss = F.mse_loss(align_outputs_2, target)
        
        return alpha * loss

    def calculate_noisy_attention_loss(self, noisy_code_attention, noisy_nl_attention):
        # 目标是最小化 noisy token 的 attention
        return (torch.sum(noisy_code_attention) + torch.sum(noisy_nl_attention))/2
    
    def focal_attention_loss(self, positive_attention, negative_attention, gamma=2.0):
        # Compute the focal loss for positive and negative attention scores
        positive_loss = (1 - positive_attention) ** gamma * positive_attention.log()
        negative_loss = negative_attention ** gamma * (1 - negative_attention).log()
        return -(positive_loss - negative_loss).mean()
    
    def dynamic_weighting_attention_loss(self, positive_attention, negative_attention):
        pos_weight = (1 - positive_attention).mean()
        neg_weight = negative_attention.mean()
        loss = pos_weight * positive_attention.log() + neg_weight * (1 - negative_attention).log()
        return -loss.mean()
    
    def regularization_attention_loss(self, positive_attention, negative_attention, alpha=0.1):
        # L1正则化
        positive_reg_loss = alpha * positive_attention.norm(1)
        negative_reg_loss = (1 - alpha) * negative_attention.norm(1)
        return positive_reg_loss + negative_reg_loss
    

    def new_alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, total_code_tokens):
        bs = code_inputs.shape[0]
        code_vec = code_outputs[1]
        nl_vec = nl_outputs[1]

        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
        loss_fct = CrossEntropyLoss()
        retrieval_loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))

        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]

        lcs_pairs = sample_align
        loss_align_code = self.build_contrastive_pairs_effecient(align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens)
        # print("loss_align_code",loss_align_code)
        
        # attention loss part
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[local_index, :, 0, :].mean(dim=0)
        
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[local_index, :, 0, :].mean(dim=0)

        # 获取对齐的 indices
        alignment_code_indices = []
        alignment_nl_indices = []

        for n, m in lcs_pairs:
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    alignment_code_indices.extend(range(m[i], m[i + 1] + 1))
            else:
                alignment_code_indices.append(m)
            
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    alignment_nl_indices.extend(range(n[i], n[i + 1] + 1))
            else:
                alignment_nl_indices.append(n)

        # 去重处理并 +1
        alignment_code_indices = list(set(alignment_code_indices))
        alignment_nl_indices = list(set(alignment_nl_indices))
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices]

        # 计算 attention loss
        epsilon = 1e-8
        # 直接使用 mask 对 positive 和 negative 的 attention 进行区分
        code_mask = torch.zeros_like(code_cls_attention, dtype=torch.bool)
        nl_mask = torch.zeros_like(nl_cls_attention, dtype=torch.bool)
        
        # 对于正样本位置，设置 mask
        code_mask[alignment_code_indices] = 1
        nl_mask[alignment_nl_indices] = 1

        # 计算正样本和负样本的 attention loss
        positive_code_attention = code_cls_attention[code_mask]
        negative_code_attention = code_cls_attention[~code_mask]
        positive_nl_attention = nl_cls_attention[nl_mask]
        negative_nl_attention = nl_cls_attention[~nl_mask]

        # 计算 attention loss
        attention_loss = (torch.sum(-torch.log(positive_code_attention + epsilon)) +
                          torch.sum(-torch.log(1.0 - negative_code_attention + epsilon)) +
                          torch.sum(-torch.log(positive_nl_attention + epsilon)) +
                          torch.sum(-torch.log(1.0 - negative_nl_attention + epsilon)))
        attention_loss = attention_loss / (len(alignment_code_indices) + len(alignment_nl_indices))
        # print("attention_loss",attention_loss)
        # print("nl_cls_attention",nl_cls_attention)

        return retrieval_loss, code_vec, nl_vec, loss_align_code, attention_loss
    
    def batch_alignment_v0(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, total_code_tokens):
        
        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]

        lcs_pairs = sample_align
        loss_align_code = self.build_contrastive_pairs(align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens)
        # print("loss_align_code",loss_align_code)
        
        # attention loss part
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[local_index, :, 0, :].mean(dim=0)
        
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[local_index, :, 0, :].mean(dim=0)

        # 获取对齐的 indices
        alignment_code_indices = []
        alignment_nl_indices = []

        for n, m in lcs_pairs:
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    alignment_code_indices.extend(range(m[i], m[i + 1] + 1))
            else:
                alignment_code_indices.append(m)
            
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    alignment_nl_indices.extend(range(n[i], n[i + 1] + 1))
            else:
                alignment_nl_indices.append(n)

        # 去重处理并 +1
        alignment_code_indices = list(set(alignment_code_indices))
        alignment_nl_indices = list(set(alignment_nl_indices))
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices]

        # 计算 attention loss
        epsilon = 1e-8
        # 直接使用 mask 对 positive 和 negative 的 attention 进行区分
        code_mask = torch.zeros_like(code_cls_attention, dtype=torch.bool)
        nl_mask = torch.zeros_like(nl_cls_attention, dtype=torch.bool)
        
        # 对于正样本位置，设置 mask
        code_mask[alignment_code_indices] = 1
        nl_mask[alignment_nl_indices] = 1

        # 计算正样本和负样本的 attention loss
        positive_code_attention = code_cls_attention[code_mask]
        negative_code_attention = code_cls_attention[~code_mask]
        positive_nl_attention = nl_cls_attention[nl_mask]
        negative_nl_attention = nl_cls_attention[~nl_mask]

        # 计算 attention loss
        attention_loss = (torch.sum(-torch.log(positive_code_attention + epsilon)) +
                          torch.sum(-torch.log(1.0 - negative_code_attention + epsilon)) +
                          torch.sum(-torch.log(positive_nl_attention + epsilon)) +
                          torch.sum(-torch.log(1.0 - negative_nl_attention + epsilon)))
        attention_loss = attention_loss / (len(alignment_code_indices) + len(alignment_nl_indices))
        # print("attention_loss",attention_loss)
        # print("nl_cls_attention",nl_cls_attention)

        return loss_align_code, attention_loss
    
    def batch_alignment(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, total_code_tokens, total_comment_tokens):
        # 使用第二层最后的隐藏状态进行对齐
        align_outputs_1 = nl_outputs[2][-1][local_index]
        align_outputs_2 = code_outputs[2][-1][local_index]
        
        code_last_layer_attention = code_outputs[3][-1]  # [batch, num_heads, seq_len, seq_len]
        nl_last_layer_attention = nl_outputs[3][-1]
        
        # 获取所有头的平均注意力分数
        code_attention = code_last_layer_attention[local_index].mean(dim=0)  # [seq_len, seq_len]
        nl_attention = nl_last_layer_attention[local_index].mean(dim=0)
        
        # code_cls_attention = code_attention[0]
        # nl_cls_attention = nl_attention[0]
        
        # 计算attention loss的部分
        alignment_code_indices = set()
        alignment_nl_indices = set()
        
        # 统计每对pairs中的token数量
        pair_token_counts = []
        pair_indices = []  # 存储每对的具体索引

        for n, m in sample_align:
            code_indices = []
            nl_indices = []
            
            if isinstance(m, list):
                for i in range(0, len(m), 2):
                    code_indices.extend(range(m[i], m[i+1] + 1))
                    alignment_code_indices.update(range(m[i], m[i+1] + 1))
            else:
                code_indices.append(m)
                alignment_code_indices.add(m)
                
            if isinstance(n, list):
                for i in range(0, len(n), 2):
                    nl_indices.extend(range(n[i], n[i+1] + 1))
                    alignment_nl_indices.update(range(n[i], n[i+1] + 1))
            else:
                nl_indices.append(n)
                alignment_nl_indices.add(n)
                
            pair_token_counts.append((len(nl_indices), len(code_indices)))
            pair_indices.append((nl_indices, code_indices))

        # 转换为tensor并加1(因为CLS token)
        alignment_code_indices = torch.tensor([i + 1 for i in alignment_code_indices], device=code_inputs.device)
        alignment_nl_indices = torch.tensor([i + 1 for i in alignment_nl_indices], device=code_inputs.device)
        
        # Calculate thresholds for aligned and non-aligned tokens
        code_aligned_threshold = 1.0 / len(alignment_code_indices)
        nl_aligned_threshold = 1.0 / len(alignment_nl_indices)
        code_other_threshold = 1.0 / total_code_tokens
        nl_other_threshold = 1.0 / total_comment_tokens

        # 使用陡峭的sigmoid函数将attention scores转换为接近0/1的值
        steepness = 50.0  # 控制sigmoid的陡峭程度
        steepness_code = 50.0  # 控制sigmoid的陡峭程度
        
        # Create masks for aligned indices
        code_aligned_mask = torch.zeros(len(code_attention[0]), dtype=torch.bool, device=code_inputs.device)
        nl_aligned_mask = torch.zeros(len(nl_attention[0]), dtype=torch.bool, device=code_inputs.device)
        code_aligned_mask[alignment_code_indices] = True
        nl_aligned_mask[alignment_nl_indices] = True
        
        # Vectorized computation for code attention
        code_thresholds = torch.where(code_aligned_mask, 
                                    code_aligned_threshold, 
                                    code_other_threshold)
        code_cls_attention = torch.sigmoid(steepness_code * (code_attention[0] - code_thresholds))
        
        # Vectorized computation for nl attention  
        nl_thresholds = torch.where(nl_aligned_mask,
                                  nl_aligned_threshold,
                                  nl_other_threshold)
        nl_cls_attention = torch.sigmoid(steepness * (nl_attention[0] - nl_thresholds))

        epsilon = 1e-7

        # 创建target attention分布 - 标注过的位置为1,其他位置为0
        code_target_attention = torch.zeros_like(code_cls_attention, device=code_inputs.device)
        nl_target_attention = torch.zeros_like(nl_cls_attention, device=code_inputs.device)
        
        # 将标注过的位置设为1
        for i in alignment_code_indices:
            code_target_attention[i] = 1.0
        for i in alignment_nl_indices:
            nl_target_attention[i] = 1.0

        # 裁剪避免数值不稳定
        # code_cls_attention = torch.clamp(code_cls_attention, min=epsilon, max=1.0-epsilon)
        # nl_cls_attention = torch.clamp(nl_cls_attention, min=epsilon, max=1.0-epsilon)
        # Calculate average threshold for code
        avg_code_threshold = (code_aligned_threshold + code_other_threshold) / 2
        
        # Create binary mask based on threshold comparison
        code_threshold_mask = (code_attention[0] > avg_code_threshold).float()

        # Calculate average threshold for nl
        avg_nl_threshold = (nl_aligned_threshold + nl_other_threshold) / 2
        
        # Create binary mask based on threshold comparison for nl
        nl_threshold_mask = (nl_attention[0] > avg_nl_threshold).float()
        
        # Get new CLS representations for code and nl
        code_cls_new, code_cls_probs, code_cls_scores = self.cls_attention(
            cls_token=align_outputs_2[0],  # Take CLS token
            sequence_tokens=align_outputs_2, # Take sequence tokens
            attention_mask=code_threshold_mask
        )
        
        nl_cls_new, nl_cls_probs, nl_cls_scores = self.cls_attention(
            cls_token=align_outputs_1[0],
            sequence_tokens=align_outputs_1,
            attention_mask=nl_threshold_mask
        )
        
        print("Average code threshold:", avg_code_threshold)
        print("Threshold comparison mask:", code_threshold_mask)
        print(code_attention[0])
        print(code_cls_attention)
        print(code_target_attention)

        # 计算二元交叉熵损失
        code_loss = -(code_target_attention * torch.log(code_cls_attention + epsilon) + 
                    (1 - code_target_attention) * torch.log(1 - code_cls_attention + epsilon))

        nl_loss = -(nl_target_attention * torch.log(nl_cls_attention + epsilon) + 
                    (1 - nl_target_attention) * torch.log(1 - nl_cls_attention + epsilon))
        # code_loss = F.binary_cross_entropy_with_logits(code_cls_attention, code_target_attention)
        # nl_loss = F.binary_cross_entropy_with_logits(nl_cls_attention, nl_target_attention)
        print(code_loss)
        # print(code_threshold)
        # aaa

        code_loss = code_loss.mean()
        nl_loss = nl_loss.mean()

        # 计算每对的attention和的差异
        pair_attention_loss = 0.0
        for nl_indices, code_indices in pair_indices:
            # 将索引加1(因为CLS token)
            nl_indices = [i + 1 for i in nl_indices]
            code_indices = [i + 1 for i in code_indices]
            
            # 计算当前pair中nl和code的attention和
            nl_attention_sum = nl_cls_attention[nl_indices].sum()
            code_attention_sum = code_cls_attention[code_indices].sum()
            
            # 计算两者差异的MSE损失
            pair_diff = (nl_attention_sum - code_attention_sum) ** 2
            pair_attention_loss += pair_diff

        # 检查损失值是否为nan
        total_elements = len(alignment_code_indices) + len(alignment_nl_indices)
        if total_elements > 0:
            attention_loss = code_loss + nl_loss + pair_attention_loss * 2
            # 如果出现nan,返回一个小的常数值
            attention_loss = torch.where(torch.isnan(attention_loss),
                               torch.tensor(1e-5, device=attention_loss.device),
                               attention_loss)
        else:
            attention_loss = torch.tensor(0.0).to(code_inputs.device)
            
        
        lcs_pairs = sample_align
        loss_align_code = self.build_contrastive_pairs_effecient(
            align_outputs_1, 
            align_outputs_2, 
            lcs_pairs, 
            total_code_tokens,
            code_cls_attention,
            nl_cls_attention
        )
        # 如果loss是nan就用一个很小的值代替
        if torch.isnan(loss_align_code):
            loss_align_code = torch.tensor(1e-8, device=loss_align_code.device)
        
        # attention_loss = code_loss + nl_loss

        return loss_align_code, attention_loss, code_cls_new, nl_cls_new
    
    def batch_alignment_with_max_pair(self, code_inputs, code_outputs, nl_outputs, local_index, sample_align, total_code_tokens):
        # Get embeddings and attention scores
        align_outputs_1 = nl_outputs[2][-1][local_index]
        align_outputs_2 = code_outputs[2][-1][local_index]
        
        code_last_layer_attention = code_outputs[3][-1]
        nl_last_layer_attention = nl_outputs[3][-1]
        
        code_attention = code_last_layer_attention[local_index].mean(dim=0)
        nl_attention = nl_last_layer_attention[local_index].mean(dim=0)
        
        code_cls_attention = code_attention[0]
        nl_cls_attention = nl_attention[0]

        # Calculate alignment loss and get max contributing pair
        loss_align_code, max_pair = self.build_contrastive_pairs_with_contribution(
            align_outputs_1,
            align_outputs_2, 
            sample_align,
            total_code_tokens,
            code_cls_attention,
            nl_cls_attention
        )

        return loss_align_code, max_pair

    def build_contrastive_pairs_with_contribution(self, align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens, code_attention_cls, nl_attention_cls):
        total_loss = 0
        max_contribution = float('-inf')
        max_pair = None
        temperature = 0.2
        temperature_pos = 0.1

        for pair in lcs_pairs:
            # Get indices for current pair
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))

            comment_indices = torch.tensor(comment_indices, device=align_outputs_1.device)
            code_indices = torch.tensor(code_indices, device=align_outputs_2.device)
            
            comment_embeddings = align_outputs_1[comment_indices]
            code_embeddings = align_outputs_2[code_indices]
            
            attention_weights = nl_attention_cls[comment_indices]
            code_attention_weights = code_attention_cls[code_indices]
            attention_coef = attention_weights.unsqueeze(1) * code_attention_weights.unsqueeze(0)
            
            # Calculate positive similarities
            pos_similarities = F.cosine_similarity(
                comment_embeddings.unsqueeze(1),
                code_embeddings.unsqueeze(0),
                dim=2
            )

            # Get negative samples
            negative_indices = list(set(range(total_code_tokens)) - set(code_indices.tolist()))
            if len(negative_indices) == 0:
                continue
                
            negative_indices = torch.tensor(negative_indices, device=align_outputs_2.device)
            negative_embeddings = align_outputs_2[negative_indices]
            
            raw_neg_similarities = F.cosine_similarity(
                comment_embeddings.unsqueeze(1),
                negative_embeddings.unsqueeze(0),
                dim=2
            )
            
            neg_similarities = 2 * torch.clamp(raw_neg_similarities, min=0) - 1
            
            pos_exp_sim = torch.exp(pos_similarities / temperature_pos)
            neg_exp_sim = torch.exp(neg_similarities / temperature)
            
            neg_similarity_sum = neg_exp_sim.sum(dim=1)
            
            denominator = pos_exp_sim + neg_similarity_sum.unsqueeze(1) + 1e-8
            nt_xent_loss = -torch.log(pos_exp_sim / denominator)
            
            attention_weight = 1.0 / (attention_coef + 1e-8)
            weighted_loss = (nt_xent_loss * attention_weight).mean()
            
            neg_loss = torch.mean(torch.clamp(-raw_neg_similarities, min=0))
            
            pair_contribution = (weighted_loss + 0.2 * neg_loss)
            total_loss += pair_contribution

            # Track maximum contributing pair
            if pair_contribution > max_contribution:
                max_contribution = pair_contribution
                # Store the first token indices of the pair
                max_pair = (comment_indices[0].item()-1, code_indices[0].item()-1)

        return total_loss, max_pair
    
    def get_new_cls_tokens(self, code_outputs, nl_outputs, local_index, total_code_tokens, total_comment_tokens):
        # Get last layer hidden states and attention weights
        align_outputs_1 = nl_outputs[2][-1][local_index]  # NL hidden states
        align_outputs_2 = code_outputs[2][-1][local_index]  # Code hidden states
        
        code_last_layer_attention = code_outputs[3][-1]  # [batch, num_heads, seq_len, seq_len]
        nl_last_layer_attention = nl_outputs[3][-1]
        
        # Get average attention scores across all heads
        code_attention = code_last_layer_attention[local_index].mean(dim=0)  # [seq_len, seq_len]
        nl_attention = nl_last_layer_attention[local_index].mean(dim=0)

        # Calculate average thresholds
        epsilon = 5e-3  # Small constant for numerical stability
        code_threshold = 1.0 / total_code_tokens + epsilon
        nl_threshold = 1.0 / total_comment_tokens + epsilon
        
        # Create binary masks based on threshold comparison
        code_threshold_mask = (code_attention[0] > code_threshold).float()
        nl_threshold_mask = (nl_attention[0] > nl_threshold).float()
        
        # Get new CLS representations using attention layer
        code_cls_new, code_cls_probs, code_cls_scores = self.cls_attention(
            cls_token=align_outputs_2[0],  # Take CLS token
            sequence_tokens=align_outputs_2,  # Take sequence tokens
            attention_mask=code_threshold_mask
        )
        
        nl_cls_new, nl_cls_probs, nl_cls_scores = self.cls_attention(
            cls_token=align_outputs_1[0],
            sequence_tokens=align_outputs_1,
            attention_mask=nl_threshold_mask
        )
        
        return code_cls_new, nl_cls_new
    
    def get_new_cls_tokens_for_code_or_nl(self, code_outputs, local_index, total_code_tokens):
        # Get last layer hidden states and attention weights
        align_outputs_2 = code_outputs[2][-1][local_index]  # Code hidden states
        
        code_last_layer_attention = code_outputs[3][-1]  # [batch, num_heads, seq_len, seq_len]

        # Get average attention scores across all heads
        code_attention = code_last_layer_attention[local_index].mean(dim=0)  # [seq_len, seq_len]

        # Calculate average thresholds
        epsilon = 5e-3  # Small constant for numerical stability
        code_threshold = 1.0 / total_code_tokens + epsilon
        
        # Create binary masks based on threshold comparison
        code_threshold_mask = (code_attention[0] > code_threshold).float()
        
        # Get new CLS representations using attention layer
        code_cls_new, code_cls_probs, code_cls_scores = self.cls_attention(
            cls_token=align_outputs_2[0],  # Take CLS token
            sequence_tokens=align_outputs_2,  # Take sequence tokens
            attention_mask=code_threshold_mask
        )
        
        return code_cls_new
    
    def build_contrastive_pairs_effecient(self, align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens, code_attention_cls, nl_attention_cls):
        loss_align_code = 0
        num_pair = 0
        temperature = 0.2
        temperature_pos = 0.1

        for pair in lcs_pairs:
            # 获取当前pair的comment和code索引
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))

            comment_indices = torch.tensor(comment_indices, device=align_outputs_1.device)
            code_indices = torch.tensor(code_indices, device=align_outputs_2.device)
            
            comment_embeddings = align_outputs_1[comment_indices]
            code_embeddings = align_outputs_2[code_indices]
            
            # 获取对应的attention weights
            attention_weights = nl_attention_cls[comment_indices]
            code_attention_weights = code_attention_cls[code_indices]
            
            # 计算注意力系数的乘积
            attention_coef = attention_weights.unsqueeze(1) * code_attention_weights.unsqueeze(0)
            
            # 计算正样本相似度
            pos_similarities = F.cosine_similarity(
                comment_embeddings.unsqueeze(1),  # [num_comments, 1, hidden_size]
                code_embeddings.unsqueeze(0),     # [1, num_codes, hidden_size]
                dim=2
            )

            negative_indices = list(set(range(total_code_tokens)) - set(code_indices.tolist()))
            if len(negative_indices) == 0:
                continue
                
            negative_indices = torch.tensor(negative_indices, device=align_outputs_2.device)
            negative_embeddings = align_outputs_2[negative_indices]
            
            # 计算负样本相似度
            raw_neg_similarities = F.cosine_similarity(
                comment_embeddings.unsqueeze(1),    # [num_comments, 1, hidden_size]
                negative_embeddings.unsqueeze(0),   # [1, num_negatives, hidden_size]
                dim=2
            )  # [num_comments, num_negatives]
            
            neg_similarities = 2 * torch.clamp(raw_neg_similarities, min=0) - 1
            
            # 应用温度系数和权重
            pos_exp_sim = torch.exp(pos_similarities / temperature_pos)
            neg_exp_sim = torch.exp(neg_similarities / temperature)
            
            # 不对正样本求和，保持每对样本的独立性
            neg_similarity_sum = neg_exp_sim.sum(dim=1)
            
            # 计算对比损失,在log外面乘上attention权重
            denominator = pos_exp_sim + neg_similarity_sum.unsqueeze(1) + 1e-8
            # 对每个正样本对分别计算损失并乘以对应的attention权重
            nt_xent_loss = -torch.log(pos_exp_sim / denominator)
            
            # 使用attention_coef的倒数作为权重，当attention乘积低时增大loss
            attention_weight = 1.0 / (attention_coef + 1e-8)  # 加上小值避免除0
            weighted_loss = (nt_xent_loss * attention_weight).mean()
            
            # 添加负样本损失
            neg_loss = torch.mean(torch.clamp(-raw_neg_similarities, min=0))
            
            loss_align_code += (weighted_loss + 0.2 * neg_loss)
            num_pair += 1

        return loss_align_code
    

      
    def build_contrastive_pairs(self, align_outputs_1, align_outputs_2, lcs_pairs, total_code_tokens, num_negative=19):
        loss_align_code = 0
        num_pair = 0

        # 遍历每一个对齐的 lcs_pairs
        for pair in lcs_pairs:
            # 获取 comment 和 code 的 indices
            comment_indices = []
            for i in range(0, len(pair[0]), 2):
                comment_indices.extend(range(pair[0][i] + 1, pair[0][i + 1] + 2))
            
            code_indices = []
            for i in range(0, len(pair[1]), 2):
                code_indices.extend(range(pair[1][i] + 1, pair[1][i + 1] + 2))

            # 遍历所有 comment indices，构建正负样本对
            # if num_pair == 0:
            for c_idx in comment_indices:
                comment_embedding = align_outputs_1[c_idx + 1]
                # 计算正样本相似度
                for code_idx in code_indices:
                    # if num_pair == 0:
                    row_similarities = []
                    row_labels = []
                    num_pair += 1
                    code_embedding = align_outputs_2[code_idx + 1]
                    pos_similarity = F.cosine_similarity(comment_embedding, code_embedding, dim=0)
                    row_similarities.append(pos_similarity.unsqueeze(0))
                    row_labels.append(torch.tensor(1, device=comment_embedding.device))  # 正样本标签为 1
                    # 构建负样本相似度（使用所有非 code_indices 的负样本）
                    negative_indices = set(range(total_code_tokens)) - set(code_indices)
                    for neg_idx in negative_indices:
                        negative_embedding = align_outputs_2[neg_idx + 1]
                        neg_similarity = F.cosine_similarity(comment_embedding, negative_embedding, dim=0)
                        row_similarities.append(neg_similarity.unsqueeze(0))
                        row_labels.append(torch.tensor(0, device=comment_embedding.device))  # 负样本标签为 0
                    temperature = 0.2
                    similarities = torch.stack(row_similarities)
                    # if num_pair == 1:
                    #     print(similarities)
                    similarities = similarities / temperature
                    exp_similarities = torch.exp(similarities)
                    # print(exp_similarities)
                    pos_similarity = exp_similarities[0]
                    neg_sum = exp_similarities[1:].sum()
                    # print(pos_similarity)
                    nt_xent_loss = -torch.log(pos_similarity / (pos_similarity + neg_sum))
                    nt_xent_loss = nt_xent_loss.mean()  # 确保是标量
                    loss_align_code += nt_xent_loss

        return loss_align_code / num_pair
    
    def contrastive_loss(self, similarities, labels, temperature=0.07):
        # 对比学习损失的实现
        similarities = similarities / temperature
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        for i in range(similarities.size(0)):
            loss = criterion(similarities[i].unsqueeze(0), labels[i].unsqueeze(0))
            total_loss += loss
        loss = total_loss / similarities.size(0)
        return loss

    def mask_tokens(self, inputs, tokenizer, mlm_probability=0.15):
        """
        Prepare masked tokens inputs for MLM, only masking specific indices if provided.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Ensure that at least one token is masked
        if masked_indices.sum() == 0:
            # If no tokens are masked, forcefully mask one token (e.g., the first one)
            masked_indices[0] = True
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # Replace the masked input tokens with the tokenizer.mask_token_id
        inputs[masked_indices] = tokenizer.mask_token_id
        
        return inputs, labels

    def compute_mlm_loss_nl(self, model, inputs, labels, origin_output):
        """
        Compute MLM loss for the given inputs and labels.
        """
        outputs = model(nl_inputs=inputs)
        # 使用最后一层隐状态
        last_hidden_state = outputs[0][0]

        # 使用 labels 作为原始向量的目标
        masked_indices = (labels != -100).nonzero(as_tuple=True)
        original_vectors = origin_output[masked_indices]
        masked_vectors = last_hidden_state[masked_indices]

        # 计算均方误差损失（MSE Loss）
        mlm_loss = F.mse_loss(masked_vectors, original_vectors)
        
        return mlm_loss

    def compute_mlm_loss_code(self, model, inputs, labels, origin_output, attn_mask, position_idx):
        """
        Compute MLM loss for code inputs and labels.
        """
        outputs = model(code_inputs=inputs,attn_mask=attn_mask,position_idx=position_idx)
        # 使用最后一层隐状态
        last_hidden_state = outputs[0][0]

        # 使用 labels 作为原始向量的目标
        masked_indices = (labels != -100).nonzero(as_tuple=True)
        original_vectors = origin_output[masked_indices]
        masked_vectors = last_hidden_state[masked_indices]

        # 计算均方误差损失（MSE Loss）
        mlm_loss = F.mse_loss(masked_vectors, original_vectors)
        
        return mlm_loss

    def batch_alignment_mlm(self, code_inputs, nl_inputs, code_outputs, nl_outputs, local_index, sample_align, total_code_tokens, tokenizer, attn_mask, position_idx, model):
        # 获取 positive pair 的索引
        alignment_code_indices_set = set()
        alignment_nl_indices_set = set()

        for n, m in sample_align:
            if isinstance(m, list):
                alignment_code_indices_set.update(chain.from_iterable(range(m[i], m[i + 1] + 1) for i in range(0, len(m), 2)))
            else:
                alignment_code_indices_set.add(m)
                
            if isinstance(n, list):
                alignment_nl_indices_set.update(chain.from_iterable(range(n[i], n[i + 1] + 1) for i in range(0, len(n), 2)))
            else:
                alignment_nl_indices_set.add(n)

        # 转换为列表并 +1
        alignment_code_indices = [idx + 1 for idx in alignment_code_indices_set]
        alignment_nl_indices = [idx + 1 for idx in alignment_nl_indices_set]

        align_outputs_1 = nl_outputs[0][local_index]
        align_outputs_2 = code_outputs[0][local_index]

        # 对注释和代码进行 mask 操作，仅对 positive pair 的 token 进行 mask
        masked_nl_inputs, nl_labels = self.mask_tokens(nl_inputs[local_index][alignment_nl_indices], tokenizer)
        masked_code_inputs, code_labels = self.mask_tokens(code_inputs[local_index][alignment_code_indices], tokenizer)

        # 计算 mlm loss
        mlm_loss_nl = self.compute_mlm_loss_nl(model, masked_nl_inputs.unsqueeze(0), nl_labels, align_outputs_1[alignment_nl_indices])
        mlm_loss_code = self.compute_mlm_loss_code(model, masked_code_inputs.unsqueeze(0), code_labels, align_outputs_2[alignment_code_indices], attn_mask[alignment_code_indices, :][:, alignment_code_indices].unsqueeze(0), position_idx[alignment_code_indices].unsqueeze(0))

        # 继续执行对齐损失和注意力损失计算
        loss_align_code = self.build_contrastive_pairs_effecient(align_outputs_1, align_outputs_2, sample_align, total_code_tokens)

        # attention loss part
        code_attentions = code_outputs[2]  # List of tensors, one for each layer
        nl_attentions = nl_outputs[2]  # List of tensors, one for each layer

        code_last_layer_attention = code_attentions[-1]
        code_cls_attention = code_last_layer_attention[local_index, :, 0, :].mean(dim=0)
        
        nl_last_layer_attention = nl_attentions[-1]
        nl_cls_attention = nl_last_layer_attention[local_index, :, 0, :].mean(dim=0)

        # 计算 attention loss
        epsilon = 1e-8
        code_mask = torch.zeros_like(code_cls_attention, dtype=torch.bool)
        nl_mask = torch.zeros_like(nl_cls_attention, dtype=torch.bool)
        
        code_mask[alignment_code_indices] = 1
        nl_mask[alignment_nl_indices] = 1

        positive_code_attention = code_cls_attention[code_mask]
        negative_code_attention = code_cls_attention[~code_mask]
        positive_nl_attention = nl_cls_attention[nl_mask]
        negative_nl_attention = nl_cls_attention[~nl_mask]

        attention_loss = (torch.sum(-torch.log(positive_code_attention + epsilon)) +
                        torch.sum(-torch.log(1.0 - negative_code_attention + epsilon)) +
                        torch.sum(-torch.log(positive_nl_attention + epsilon)) +
                        torch.sum(-torch.log(1.0 - negative_nl_attention + epsilon)))
        attention_loss = attention_loss / (len(alignment_code_indices) + len(alignment_nl_indices))

        mlm_loss = (mlm_loss_nl + mlm_loss_code) / 2

        return loss_align_code, attention_loss, mlm_loss

