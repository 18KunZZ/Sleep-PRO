import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn.cluster import KMeans


class SupSwAVLoss(nn.Module):
    """
    基于SwAV思路改进的监督对比损失函数（完整优化版）
    核心特性：向量化计算、预计算复用、设备自动同步、支持混合精度
    适用场景：图像/文本分类、特征表示学习等监督任务
    """

    def __init__(self,
                 num_classes=5,
                 num_prototypes_per_class=3,
                 feature_dim=128,  # 输入特征维度（需与模型输出特征维度一致）
                 temperature=0.1,
                 alpha=5.0,  # 标签约束惩罚系数
                 beta=0.1,  # 原型内聚正则化权重
                 sinkhorn_iter=3,
                 lambda_sym=0.5):  # Sinkhorn迭代次数（2-3次足够收敛）
        super(SupSwAVLoss, self).__init__()
        self.num_classes = num_classes
        self.K = num_classes * num_prototypes_per_class  # 总原型数
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.sinkhorn_iter = sinkhorn_iter
        self.proto_per_class = num_prototypes_per_class

        # 1. 初始化可学习原型向量（与输入特征维度匹配）
        self.prototypes = nn.Parameter(torch.randn(self.K, feature_dim))
        nn.init.kaiming_normal_(self.prototypes)  # 初始化加速收敛

        # 2. 预计算「类别→原型」的掩码（仅初始化1次，避免重复计算）
        self.register_buffer(
            "class_proto_mask",
            self._precompute_class_proto_mask()  # [num_classes, K]，bool类型
        )

    def _precompute_class_proto_mask(self):
        """预计算每个类别对应的原型位置掩码（向量化标签约束的核心）"""
        mask = torch.zeros(self.num_classes, self.K, dtype=torch.bool)
        for c in range(self.num_classes):
            start = c * self.proto_per_class
            end = start + self.proto_per_class
            mask[c, start:end] = True  # 标记该类别对应的原型位置
        return mask

    def sinkhorn_knopp(self, matrix, row_sum=1.0, col_sum=None):
        """优化版Sinkhorn-Knopp算法：in-place操作+数值稳定"""
        n, m = matrix.shape
        col_sum = row_sum if col_sum is None else col_sum

        # 提前计算指数并限制数值范围（避免下溢/溢出）
        matrix = torch.exp(matrix).clamp(min=1e-12, max=1e12)

        for _ in range(self.sinkhorn_iter):
            # 行归一化（in-place操作减少内存开销）
            row_sums = matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
            matrix.div_(row_sums).mul_(row_sum)

            # 列归一化（in-place操作）
            col_sums = matrix.sum(dim=0, keepdim=True).clamp(min=1e-8)
            matrix.div_(col_sums).mul_(col_sum)

        return matrix

    def forward(self, features, labels):
        """
        前向计算（支持混合精度训练，自动适配CPU/GPU）
        Args:
            features: 输入特征，形状为 [batch_size, num_views, feature_dim]
                      （num_views：多视图数量，如2个增强视图）
            labels: 样本标签，形状为 [batch_size]（0~num_classes-1的整数）
        Returns:
            total_loss: 总损失（交换预测损失 + 原型内聚正则化损失）
        """
        # 1. 自动同步设备（原型、掩码与输入特征保持一致）
        device = features.device
        if self.prototypes.device != device:
            self.prototypes.data = self.prototypes.data.to(device)
            self.class_proto_mask = self.class_proto_mask.to(device)

        # 2. 解析输入维度
        batch_size, num_views, _ = features.shape
        if features.shape[2] != self.feature_dim:
            raise ValueError(f"特征维度不匹配！输入{features.shape[2]}，预期{self.feature_dim}")

        # 3. 预计算所有视图的「特征-原型相似度矩阵」（减少冗余计算）
        # 视图维度转置：[num_views, batch_size, feature_dim]
        view_features = features.permute(1, 0, 2)
        # 一次性计算所有视图的相似度：[num_views, batch_size, K]
        view_sim_matrices = torch.matmul(view_features, self.prototypes.T)

        # 4. 向量化计算原型内聚正则化损失（消除类别级循环）
        # 按类别分组原型：[num_classes, proto_per_class, feature_dim]
        class_protos = self.prototypes.view(self.num_classes, self.proto_per_class, -1)
        # 计算每个类别的原型均值：[num_classes, 1, feature_dim]
        proto_means = class_protos.mean(dim=1, keepdim=True)
        # 向量化计算方差和（所有类别一次性处理）
        proto_cohesion_loss = torch.mean((class_protos - proto_means) ** 2)

        # 5. 向量化计算交换预测损失（消除样本级循环）
        total_swap_loss = 0.0
        # 遍历所有视图组合（i=标签方，j=预测方，i≠j）
        for i in range(num_views):
            for j in range(num_views):
                if i == j:
                    continue

                # 5.1 生成当前视图i的相似度矩阵（带标签约束）
                # 所有样本的有效原型掩码：[batch_size, K]
                valid_proto_mask = self.class_proto_mask[labels]
                # 生成惩罚矩阵（无效原型减alpha，有效原型无惩罚）
                penalty = torch.full_like(view_sim_matrices[i], -self.alpha, device=device)
                penalty[valid_proto_mask] = 0.0  # 有效原型位置惩罚置0
                sim_matrix_i = view_sim_matrices[i] + penalty  # [batch_size, K]

                # 5.2 生成软分配码Q（标签方）
                Q = self.sinkhorn_knopp(sim_matrix_i)  # [batch_size, K]

                # 5.3 生成预测分配码P（预测方）
                sim_matrix_j = view_sim_matrices[j]  # [batch_size, K]
                P = F.softmax(sim_matrix_j / self.temperature, dim=1)  # [batch_size, K]

                # 5.4 计算交叉熵损失（向量化，避免样本循环）
                swap_loss = -torch.mean(torch.sum(Q * torch.log(P + 1e-8), dim=1))
                total_swap_loss += swap_loss


        # 6. 计算总损失（平均多视图组合损失 + 正则化损失）
        avg_swap_loss = total_swap_loss / (num_views * (num_views - 1))
        total_loss = avg_swap_loss + self.beta * proto_cohesion_loss

        return total_loss
