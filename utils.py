import os
import random
import numpy as np
import torch
import dgl
import logging
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
from math import sqrt
from scipy import stats

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


'''def graph_collate_func(x):
    d,d_e, p, y = zip(*x)
    d_e = dgl.batch(d_e)
    # return d, torch.tensor(np.array(p)), torch.tensor(y)
    return d,d_e, p, torch.tensor(y)'''
def graph_collate_func(x):
    """
    Collate function for batching drug and protein features along with graph representations.
    
    Args:
        x (list of tuples): Each tuple contains (d, d_e, p, y) where:
            - d: Precomputed drug feature (Tensor or array).
            - d_e: Drug graph representation (DGLGraph).
            - p: Precomputed protein feature (Tensor or array).
            - y: Label (float or int).
    
    Returns:
        Tuple of batched data:
            - d (Tensor): Batched drug features.
            - d_e (DGLGraph): Batched drug graph representations.
            - p (Tensor): Batched protein features.
            - y (Tensor): Batched labels.
    """
    # Unpack the input data
    d, d_e,p,p_2,p_mask,y = zip(*x)
    
    # d = torch.tensor(np.array(d), dtype=torch.float32)
    # Batch the graph data for drugs
    d_e = dgl.batch(d_e)
    # Convert protein features to a tensor
    # p_1 = torch.tensor(np.array(p_1), dtype=torch.float32)
    p = torch.tensor(np.array(p), dtype=torch.float32)
    p_mask = torch.tensor(np.array(p_mask), dtype=torch.float32)
    p_2 = torch.tensor(np.array(p_2), dtype=torch.float32)
    # Convert labels to a tensor
    y = torch.tensor(y, dtype=torch.float32)
    
    return d, d_e,p,p_2, p_mask,y

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if hasattr(dataset, '_lazy_init_h5'):
        dataset._lazy_init_h5()


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_layer = nn.Linear(feature_dim, feature_dim, bias=True)

    def forward(self, features):
        attention_scores = self.attention_layer(features)  
        attention_scores = torch.tanh(attention_scores)  # 可选激活函数
        attention_weights = F.softmax(attention_scores, dim=1)  # 对序列长度 L 归一化

        # 加权特征 [B, L, d]
        attended_features = features * attention_weights

        return attended_features, attention_weights


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        # 使用一个线性层来调整维度（128 * 2 * 256 -> 256）
        self.combine = nn.Linear(256 * 2, 256)

    def forward(self, LM_fea, Sty_fea):
        # 确保输入的形状是 (64, 128, 256)
        batch_size, seq_len, feature_dim = LM_fea.shape

        # 特征拼接（dim=2）
        fused_fea = torch.cat((LM_fea, Sty_fea), dim=2)  # (64, 128, 256 * 2)

        # 使用Linear层来处理拼接后的特征并调整维度
        fused_fea = self.combine(fused_fea.view(batch_size, seq_len, -1))  # (64, 128, 512) -> (64, 128, 256)

        return fused_fea

class Cross_WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 映射到相同的维度
        self.qkv_1 = nn.Linear(512, dim * 3, bias=qkv_bias)  # 药物特征
        self.qkv_2 = nn.Linear(1000, dim * 3, bias=qkv_bias)  # 蛋白质特征

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        x: 药物特征
        y: 蛋白质特征
        """
        B_, N, C = x.shape
        qkv1 = self.qkv_1(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv_2(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        q = q1
        k = k1 + k2  # 合并蛋白质和药物的k
        v = v1 + v2  # 合并蛋白质和药物的v

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 位置编码的处理
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class GLTB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GLTB, self).__init__()
        # Local Branch (Convolution)
        self.local_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.local_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.local_bn1 = nn.BatchNorm1d(out_channels)
        self.local_bn2 = nn.BatchNorm1d(out_channels)

        # Global Branch (Attention Mechanism)
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)

        # Final Fusion Convolution (Long Convolution)
        self.fusion_conv = nn.Conv1d(out_channels * 2, out_channels, kernel_size=7, padding=3)  # 假设长卷积使用7x7

    def forward(self, x_drug, x_protein):
        # Local Branch Processing
        local_drug = self.local_branch(x_drug)
        local_protein = self.local_branch(x_protein)

        # Global Branch Processing (Attention Mechanism)
        att_drug = self.attention_mechanism(x_drug, x_protein)
        att_protein = self.attention_mechanism(x_protein, x_drug)

        # Feature Fusion of Local and Global
        fused_drug = torch.cat((local_drug, att_drug), dim=2).transpose(1, 2)
        fused_protein = torch.cat((local_protein, att_protein), dim=2).transpose(1, 2)

        # Apply Long Convolution on Fused Features
        out_drug = self.fusion_conv(fused_drug).transpose(1, 2)
        out_protein = self.fusion_conv(fused_protein).transpose(1, 2)

        # Final Output
        out = torch.cat((out_drug, out_protein), dim=1)
        return out

    def local_branch(self, x):
        local = self.local_bn2(F.relu(self.local_conv2(self.local_bn1(F.relu(self.local_conv1(x.transpose(1, 2)))))))
        return local.transpose(1, 2)

    def attention_mechanism(self, query_input, key_value_input):
        query = self.query(query_input)
        key = self.key(key_value_input)
        value = self.value(key_value_input)

        attention_weights = self.softmax(torch.matmul(query, key.transpose(-2, -1)))
        attention_output = torch.matmul(attention_weights, value)
        return attention_output

class ProbAttention(nn.Module): #概率注意力机制
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):#主要用来计算查询（Q）和键（K）之间的相似度
        # Q [B, H, L, D]
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        indx_sample = torch.randint(L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), indx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):#根据值（V）生成初步的上下文（context）。如果不使用掩码，context 就是值（V）的求和；如果使用掩码，则生成累积的上下文。
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):#根据给定的注意力分数更新上下文信息
        B, H, L_V, D = V.shape

        if self.mask_flag:#如果需要掩码，就会先根据给定的掩码对注意力分数进行屏蔽，再计算加权的值（V）并更新上下文。
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V)
        return context_in

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        queries = queries.view(B, H, L, -1)
        keys = keys.view(B, H, S, -1)
        values = values.view(B, H, S, -1)

        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()

        scores_top, index = self._prob_QK(queries, keys, u, U)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L, attn_mask)

        return context.contiguous()


class AttentionLayer(nn.Module):#实现了 多头注意力机制
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # print(queries.shape)
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ).view(B, L, -1)

        return self.out_projection(out)


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def get_mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def rm2(y,f):
    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class ContrastLoss(nn.Module):
    def __init__(self, source_number):
        super(ContrastLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.source_number = source_number
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        if self.source_number > 2:
            contrast_label = contrast_label.float()
            anchor_fea = anchor_fea.detach()
            loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
            loss = loss * contrast_label
        else:
            loss = 0 * contrast_label
        return loss.mean()

class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature  # 温度参数

    def forward(self, x1d, x3d, labels):
        """
        计算 InfoNCE 损失函数。
        
        :param x1d: 1D 表示，形状为 (batch_size, embedding_dim)
        :param x3d: 3D 表示，形状为 (batch_size, embedding_dim)
        :param labels: 每个样本的标签，形状为 (batch_size,)
        :return: InfoNCE 损失值
        """
        batch_size = x1d.shape[0]

        # 计算 1D 和 3D 表示之间的余弦相似度
        sim_matrix = self.similarity(x1d, x3d)

        # 构建正负样本对
        positive_mask = self.create_positive_mask(labels, batch_size)
        negative_mask = 1 - positive_mask  # 负样本是正样本的补集

        # 正样本对的相似度
        positive_sim = sim_matrix * positive_mask

        # 负样本对的相似度
        negative_sim = sim_matrix * negative_mask

        # 计算 InfoNCE 损失
        positive_sim_sum = torch.sum(torch.exp(positive_sim / self.temperature), dim=1)
        negative_sim_sum = torch.sum(torch.exp(negative_sim / self.temperature), dim=1)

        loss = -torch.mean(torch.log(positive_sim_sum / (positive_sim_sum + negative_sim_sum)))

        return loss
    def similarity(self, x1d, x3d):
        """
        计算 1D 表示和 3D 表示之间的余弦相似度矩阵。

        :param x1d: 1D 表示，形状为 (batch_size, embedding_dim)
        :param x3d: 3D 表示，形状为 (batch_size, embedding_dim)
        :return: 余弦相似度矩阵，形状为 (batch_size, batch_size)
        """
        # 归一化表示
        x1d_normalized = F.normalize(x1d, p=2, dim=1)
        x3d_normalized = F.normalize(x3d, p=2, dim=1)

        # 计算余弦相似度
        sim_matrix = torch.matmul(x1d_normalized, x3d_normalized.T)
        return sim_matrix

    def create_positive_mask(self, labels, batch_size):
        """
        根据标签创建正样本的掩码，表示哪些样本是正样本对。

        :param labels: 样本的标签，形状为 (batch_size,)
        :param batch_size: 批次大小
        :return: 正样本掩码，形状为 (batch_size, batch_size)
        """
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        return positive_mask.float()

class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        """ 
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Key and Value should always come from the same source (Aiming to forcus on), Query comes from the other source
        Self-Att : Both three Query, Key, Value come from the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix