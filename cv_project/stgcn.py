"""
精简版 ST-GCN(适配 COCO12 自定义骨架)
- 输入: (N, C=3, T, V=12)
- 邻接矩阵: 用 pose_coco12.build_adjacency() 出来的 (12,12)
- 关键设计: extract_feature() 出 embedding,forward() 出分类 logits。
  同一个模型 —— 预训练用分类头训;建基准库 / 推理只取 embedding。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 空间图卷积 ----------
class GraphConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x, A):
        # x: (N,C,T,V),A: (V,V)
        x = self.conv(x)                          # 通道变换
        x = torch.einsum("nctv,vw->nctw", x, A)   # 沿关节维按邻接聚合
        return x


# ---------- 一个 ST-GCN block(空间图卷积 + 时间卷积 + 残差) ----------
class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, kt=9, stride=1, residual=True):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c)
        self.bn_gcn = nn.BatchNorm2d(out_c)
        pad = ((kt - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_c, out_c, (kt, 1), (stride, 1), pad),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, (stride, 1)),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x, A):
        res = self.residual(x)
        x = self.relu(self.bn_gcn(self.gcn(x, A)))
        x = self.tcn(x)
        return self.relu(x + res)


# ---------- 主模型 ----------
class STGCN(nn.Module):
    def __init__(self, A, in_channels=3, num_class=10,
                 num_joints=12, edge_importance=True):
        super().__init__()
        A = torch.as_tensor(np.asarray(A), dtype=torch.float32)
        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, residual=False),
            STGCNBlock(64, 64),
            STGCNBlock(64, 128, stride=2),     # 时间维下采样
            STGCNBlock(128, 256, stride=2),    # 再下采样
        ])

        if edge_importance:
            # 每层一个可学习的"边重要性"权重,乘在邻接矩阵上
            self.edge_imp = nn.ParameterList(
                [nn.Parameter(torch.ones_like(A)) for _ in self.blocks]
            )
        else:
            self.edge_imp = [1.0] * len(self.blocks)

        self.embed_dim = 256
        self.fc = nn.Linear(256, num_class)

    def extract_feature(self, x):
        """x: (N,C,T,V) -> embedding (N,256)"""
        N, C, T, V = x.size()
        # 数据标准化(对 C*V 做 BN)
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()   # 回到 (N,C,T,V)

        for blk, imp in zip(self.blocks, self.edge_imp):
            x = blk(x, self.A * imp)

        # 时间 + 关节维全局平均池化 -> 固定长度向量
        x = F.avg_pool2d(x, x.size()[2:]).view(N, -1)             # (N,256)
        return x

    def forward(self, x):
        """预训练用:返回分类 logits"""
        return self.fc(self.extract_feature(x))

    @torch.no_grad()
    def get_embedding(self, x, normalize=True):
        """建基准库 / 推理用:返回 embedding(默认 L2 归一化,方便 cosine 比对)"""
        self.eval()
        feat = self.extract_feature(x)
        if normalize:
            feat = F.normalize(feat, p=2, dim=1)
        return feat


# ---------- 工具:把预处理后的 clip 转成模型输入 ----------
def clips_to_tensor(clips):
    """
    clips: 单个 (T,V,C) 或一批 (N,T,V,C)  ->  (N,C,T,V) tensor
    """
    arr = np.asarray(clips, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None]                       # (T,V,C) -> (1,T,V,C)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return t


# ---------- 自测 ----------
if __name__ == "__main__":
    from pose_coco12 import build_adjacency

    A = build_adjacency()                     # (12,12)
    model = STGCN(A, num_class=10)

    # 模拟一批变长 clip,各自重采样到 300 帧后再进模型
    dummy = np.random.randn(4, 300, 12, 3).astype(np.float32)
    x = clips_to_tensor(dummy)                # (4,3,300,12)
    print("输入:", tuple(x.shape))

    logits = model(x)
    emb = model.get_embedding(x)
    print("分类 logits:", tuple(logits.shape))   # (4,10)
    print("embedding :", tuple(emb.shape))       # (4,256)
    print("embedding 已 L2 归一化,范数≈1:", float(emb[0].norm()))
