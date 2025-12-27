import torch


class RunningScale:
    """运行中的修剪尺度估计器（Running trimmed scale estimator）。"""

    def __init__(self, cfg):
        """
        初始化 RunningScale 对象。

        参数：
            cfg: 配置对象，包含超参数（例如 tau）。
        """
        self.cfg = cfg
        # 根据硬件可用性选择设备（GPU 或 CPU）
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # 初始化尺度值为 1
        self._value = torch.ones(1, dtype=torch.float32, device=self.device)
        # 初始化百分位数，用于计算修剪尺度
        self._percentiles = torch.tensor(
            [5, 95], dtype=torch.float32, device=self.device
        )

    def state_dict(self):
        """
        返回当前对象的状态字典。

        返回：
            dict: 包含当前尺度值和百分位数的字典。
        """
        return dict(value=self._value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        """
        从状态字典加载对象状态。

        参数：
            state_dict: 包含尺度值和百分位数的字典。
        """
        self._value.data.copy_(state_dict["value"])
        self._percentiles.data.copy_(state_dict["percentiles"])

    @property
    def value(self):
        """
        返回当前尺度值。

        返回：
            float: 当前尺度值。
        """
        return self._value.cpu().item()

    def _percentile(self, x):
        """
        计算输入张量的指定百分位数。

        参数：
            x (torch.Tensor): 输入张量。

        返回：
            torch.Tensor: 百分位数值。
        """
        x_dtype, x_shape = x.dtype, x.shape
        # 将张量展平为二维
        x = x.view(x.shape[0], -1)
        # 对张量进行排序
        in_sorted, _ = torch.sort(x, dim=0)
        # 计算百分位数的位置
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)  # 向下取整
        ceiled = floored + 1  # 向上取整
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1  # 防止越界
        # 计算权重
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        # 根据权重计算插值
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        # 恢复原始形状并返回
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        """
        更新当前尺度值。

        参数：
            x (torch.Tensor): 输入张量。
        """
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self._value.data.lerp_(value, self.cfg.tau)

    def __call__(self, x, update=False):
        """
        调用对象时的行为。

        参数：
            x (torch.Tensor): 输入张量。
            update (bool): 是否更新尺度值。

        返回：
            torch.Tensor: 经过尺度调整的张量。
        """
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        """
        返回对象的字符串表示。

        返回：
            str: 对象的字符串表示。
        """
        return f"RunningScale(S: {self.value})"