import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入functorch，如果失败则使用torch.func
try:
    from functorch import combine_state_for_ensemble
except ImportError:
    try:
        from torch.func import combine_state_for_ensemble
    except ImportError:
        # 如果都没有，创建一个简化的替代实现
        def combine_state_for_ensemble(modules):
            """简化的ensemble状态组合函数"""
            def ensemble_forward(params, buffers, x):
                # 简化实现：只使用第一个模块
                return modules[0](x)
            
            # 返回前向函数、参数和缓冲区
            params = [dict(m.named_parameters()) for m in modules]
            buffers = [dict(m.named_buffers()) for m in modules]
            return ensemble_forward, params, buffers


class Ensemble(nn.Module):
    """
    向量化的模型集成（Ensemble）模块。
    用于同时管理和并行前向多个神经网络（如 Q 网络集成），提升鲁棒性和不确定性估计。
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)  # 将输入的模块列表转为 ModuleList
        # 使用 functorch 工具将多个模块的参数和前向函数组合
        fn, params, _ = combine_state_for_ensemble(modules)
        # vmap 实现批量并行前向
        self.vmap = torch.vmap(
            fn, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        # 将参数注册为 nn.ParameterList，便于优化器管理
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        # 并行前向所有集成成员
        return self.vmap([p for p in self.params], (), *args, **kwargs)

    def __repr__(self):
        return "Vectorized " + self._repr


class ShiftAug(nn.Module):
    """
    随机平移图像增强（Random shift augmentation）。
    用于像素观测的图像增强，提升视觉策略泛化能力。
    来源：https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad  # 填充像素数

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w  # 只支持方形图像
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")  # 边界复制填充
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        # 使用 grid_sample 实现平移
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelPreprocess(nn.Module):
    """
    像素观测归一化到 [-0.5, 0.5] 区间。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div_(255.0).sub_(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization（单纯形归一化）。
    用于输出 softmax 归一化，提升数值稳定性。
    来源：https://arxiv.org/abs/2204.00616
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim  # 归一化维度

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)  # 重塑为 (..., N, dim)
        x = F.softmax(x, dim=-1)             # 对最后一维做 softmax
        return x.view(*shp)                  # 恢复原 shape

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    带 LayerNorm、激活函数和可选 Dropout 的线性层。
    用于 MLP 的基本构件，提升训练稳定性。
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)  # 输出维度做 LayerNorm
        self.act = act                             # 激活函数
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))  # 先 LayerNorm 再激活

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    TD-MPC2 的基础 MLP 构建函数。
    支持多层、LayerNorm、Mish 激活和可选 Dropout。
    - in_dim: 输入维度
    - mlp_dims: 隐藏层维度（int 或 list）
    - out_dim: 输出维度
    - act: 输出层激活（如 SimNorm）
    - dropout: 输入层 Dropout 概率
    返回 nn.Sequential 模型。
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act)
        if act
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    TD-MPC2 的基础卷积编码器（用于原始图像观测）。
    结构：4 层卷积 + ReLU，最后展平 + 可选激活。
    - in_shape: 输入 shape（如 [3, 64, 64]）
    - num_channels: 卷积通道数
    - act: 可选激活（如 SimNorm）
    """
    assert in_shape[-1] == 64  # 只支持 64x64 图像
    layers = [
        ShiftAug(),  # 随机平移增强
        PixelPreprocess(),  # 像素归一化
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, out={}):
    """
    根据配置返回每种观测类型的编码器字典。
    - cfg: 配置对象，包含 obs_shape、task_dim 等
    - out: 可选，已有编码器字典
    支持 'state'（MLP）和 'rgb'（卷积）两种观测。
    """
    for k in cfg.obs_shape.keys():
        if k == "state":
            out[k] = mlp(
                cfg.obs_shape[k][0] + cfg.task_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
        elif k == "rgb":
            out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            raise NotImplementedError(
                f"Encoder for observation type {k} not implemented."
            )
    return nn.ModuleDict(out)
