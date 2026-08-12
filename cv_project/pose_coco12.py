"""
COCO12 工具模块 —— 用于 ST-GCN 动作 embedding 的跨数据源对齐
统一处理:AIST++ (COCO17, 3D) 训练数据  与  ARKit body skeleton 录制数据

核心原则:训练 / 基准录制 / 查询,三路数据全部走这同一套函数,
         保证图拓扑、归一化、朝向对齐 100% 一致。
"""

import numpy as np

# ============================================================
# 1. COCO12 关节定义(在 COCO17 基础上去掉前 5 个头部点)
# ============================================================
# COCO17 原始顺序: 0鼻 1左眼 2右眼 3左耳 4右耳
#                  5左肩 6右肩 7左肘 8右肘 9左腕 10右腕
#                  11左髋 12右髋 13左膝 14右膝 15左踝 16右踝
#
# 去掉 0~4 后,COCO12 重新编号如下:
COCO12_NAMES = [
    "L_shoulder",  # 0
    "R_shoulder",  # 1
    "L_elbow",     # 2
    "R_elbow",     # 3
    "L_wrist",     # 4
    "R_wrist",     # 5
    "L_hip",       # 6
    "R_hip",       # 7
    "L_knee",      # 8
    "R_knee",      # 9
    "L_ankle",     # 10
    "R_ankle",     # 11
]

# AIST++ 的 COCO17 数组里,这 12 个点对应的原始索引(顺序与 COCO12_NAMES 对齐)
COCO17_TO_COCO12_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


# ============================================================
# 2. ARKit body skeleton -> COCO12 映射表
# ============================================================
# 注意:这些是 ARSkeletonDefinition.defaultBody3D 的标准关节名。
# 请用你实际导出的关节名核对!尤其是 *_shoulder vs *_arm 的选择。
ARKIT_TO_COCO12 = {
    "L_shoulder": "left_arm_joint",       # 可选: left_shoulder_1_joint(更靠锁骨内侧)
    "R_shoulder": "right_arm_joint",
    "L_elbow":    "left_forearm_joint",
    "R_elbow":    "right_forearm_joint",
    "L_wrist":    "left_hand_joint",
    "R_wrist":    "right_hand_joint",
    "L_hip":      "left_upLeg_joint",
    "R_hip":      "right_upLeg_joint",
    "L_knee":     "left_leg_joint",
    "R_knee":     "right_leg_joint",
    "L_ankle":    "left_foot_joint",
    "R_ankle":    "right_foot_joint",
}


# ============================================================
# 3. COCO12 的 ST-GCN 邻接图
# ============================================================
# COCO 没有 pelvis / neck 中心点,躯干用 肩-肩 / 髋-髋 / 肩-髋 四边形近似连接。
# 边用 COCO12 的新索引表示(无向)。
COCO12_EDGES = [
    # 左臂
    (0, 2), (2, 4),
    # 右臂
    (1, 3), (3, 5),
    # 左腿
    (6, 8), (8, 10),
    # 右腿
    (7, 9), (9, 11),
    # 躯干四边形
    (0, 1),   # 左肩-右肩
    (6, 7),   # 左髋-右髋
    (0, 6),   # 左肩-左髋
    (1, 7),   # 右肩-右髋
]


def build_adjacency(num_nodes=12, edges=COCO12_EDGES, self_loop=True):
    """构造 ST-GCN 用的对称归一化邻接矩阵 A_hat = D^-1/2 (A+I) D^-1/2"""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loop:
        A += np.eye(num_nodes, dtype=np.float32)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D = np.diag(deg_inv_sqrt)
    return D @ A @ D


# ============================================================
# 4. 数据读取 / 转换
# ============================================================
def aist_to_coco12(kp17):
    """
    AIST++ 的 (T,17,3) -> (T,12,3)
    建议传入 keypoints3d_optim(更平滑)。
    """
    kp17 = np.asarray(kp17, dtype=np.float32)
    return kp17[:, COCO17_TO_COCO12_IDX, :]


def arkit_to_coco12(arkit_dict):
    """
    ARKit 录制数据 -> (T,12,3)
    arkit_dict: { joint_name: ndarray(T,3) }  每个关节的位置(模型/世界空间)。
                位置需从 ARSkeleton 各 joint 的 modelTransform 平移分量提取。
    """
    T = None
    for v in arkit_dict.values():
        T = np.asarray(v).shape[0]
        break
    out = np.zeros((T, 12, 3), dtype=np.float32)
    for coco_i, coco_name in enumerate(COCO12_NAMES):
        ark_name = ARKIT_TO_COCO12[coco_name]
        if ark_name not in arkit_dict:
            raise KeyError(f"ARKit 数据缺少关节: {ark_name},请核对导出的关节名")
        out[:, coco_i, :] = np.asarray(arkit_dict[ark_name], dtype=np.float32)
    return out


# ============================================================
# 5. 统一归一化(三路数据必须用完全相同的流程)
# ============================================================
L_HIP, R_HIP, L_SHO, R_SHO = 6, 7, 0, 1


def normalize_skeleton(seq, center=True, scale=True):
    """
    seq: (T,12,3)  ——  单位无所谓(mm 或 m),归一化后量纲被消除。
    1) 以左右髋中点(虚拟 pelvis)为原点 -> 消除全局平移
    2) 除以整段 clip 的平均躯干长度(肩中点-髋中点)-> 消除身高/单位差异
       注意用 clip 平均而非逐帧,避免尺度随帧抖动。
    """
    seq = np.asarray(seq, dtype=np.float32).copy()

    if center:
        root = (seq[:, L_HIP, :] + seq[:, R_HIP, :]) / 2.0   # (T,3)
        seq = seq - root[:, None, :]

    if scale:
        sho_mid = (seq[:, L_SHO, :] + seq[:, R_SHO, :]) / 2.0
        hip_mid = (seq[:, L_HIP, :] + seq[:, R_HIP, :]) / 2.0
        torso = np.linalg.norm(sho_mid - hip_mid, axis=1)    # (T,)
        torso_len = float(np.mean(torso[torso > 1e-6]))
        if torso_len > 1e-6:
            seq = seq / torso_len

    return seq


def align_orientation_first_frame(seq):
    """
    First-frame 朝向归一化:用第一帧把人摆正到 canonical 朝向,
    整段 clip 应用同一旋转 —— 保留转身动作(舞蹈关键),只消除起始朝向差异。

    canonical: X = 左髋->右髋方向, Y = 髋中点->肩中点(向上), Z = X×Y(前方)
    """
    seq = np.asarray(seq, dtype=np.float32).copy()
    f0 = seq[0]  # (12,3)

    x_axis = f0[R_HIP] - f0[L_HIP]
    y_tmp  = (f0[L_SHO] + f0[R_SHO]) / 2.0 - (f0[L_HIP] + f0[R_HIP]) / 2.0

    x = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    z = np.cross(x, y_tmp); z = z / (np.linalg.norm(z) + 1e-8)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=0)   # 世界 -> canonical 的旋转 (3,3)
    return seq @ R.T                  # 对每帧每点应用旋转


def resample_time(seq, target_len=300):
    """
    时间重采样:把任意帧数的 clip 等比例插值到固定 target_len 帧。
    保留动作相对节奏(不是粗暴截断),让所有 embedding 在同一时间尺度下可比。
    seq: (T,V,C) -> (target_len,V,C)
    """
    seq = np.asarray(seq, dtype=np.float32)
    T, V, C = seq.shape
    if T == target_len:
        return seq
    old = np.linspace(0.0, 1.0, T)
    new = np.linspace(0.0, 1.0, target_len)
    flat = seq.reshape(T, V * C)
    out = np.empty((target_len, V * C), dtype=np.float32)
    for k in range(V * C):
        out[:, k] = np.interp(new, old, flat[:, k])
    return out.reshape(target_len, V, C)


# ============================================================
# 6. 完整预处理入口(三路统一调用)
# ============================================================
def preprocess(seq_coco12, do_orientation=True, target_len=300):
    """
    clip 级别预处理:朝向对齐 -> 居中+尺度归一化 -> 时间重采样到固定长度。
    AIST 预训练 / ARKit 建基准 / ARKit 推理,三路必须用完全相同的参数。
    target_len=None 则不重采样(保持原帧数)。
    """
    if do_orientation:
        seq_coco12 = align_orientation_first_frame(seq_coco12)
    seq_coco12 = normalize_skeleton(seq_coco12)
    if target_len is not None:
        seq_coco12 = resample_time(seq_coco12, target_len)
    return seq_coco12


# ============================================================
# 用法示例
# ============================================================
if __name__ == "__main__":
    import pickle

    # --- 训练侧:AIST++ ---
    with open("gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb") as fp:
        d = pickle.load(fp)
    kp12 = aist_to_coco12(d["keypoints3d_optim"])     # (720,12,3)
    train_clip = preprocess(kp12)

    # --- 推理侧:ARKit(示意,arkit_dict 由你的录制导出)---
    # arkit_dict = {"left_arm_joint": np.zeros((300,3)), ...}
    # query_clip = preprocess(arkit_to_coco12(arkit_dict))

    A_hat = build_adjacency()   # 喂给 ST-GCN 的邻接矩阵
    print("clip shape:", train_clip.shape, " A_hat shape:", A_hat.shape)
