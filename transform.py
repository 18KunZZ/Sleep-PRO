import torch
import random
import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.ndimage.interpolation import shift


class TwoTransform:
    
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


####################################################################################
# class MultiCropTransform:
#     def __init__(self, long_transform, short_transform, num_short=4,
#                  long_crop_ratio=0.5, short_crop_ratio=0.2, sr=100,
#                  # -------------------------- 新增：统一视图的目标长度（按短裁剪长度对齐） --------------------------
#                  target_length_mode="short_crop"):  # "short_crop"=按短裁剪长度统一，"fixed"=自定义固定长度
#         """
#         多裁剪变换类，实现长裁剪和短裁剪功能（适配SupConLoss版）
#
#         参数:
#             long_transform: 长裁剪的增强变换
#             short_transform: 短裁剪的增强变换
#             num_short: 短裁剪的数量
#             long_crop_ratio: 长裁剪占原始长度的比例（0-1之间）
#             short_crop_ratio: 短裁剪占原始长度的比例（0-1之间）
#             sr: 采样率，用于计算实际时长
#             target_length_mode: 视图长度统一模式，"short_crop"=跟随短裁剪长度，"fixed"=需额外传fixed_length
#             fixed_length: 仅target_length_mode="fixed"时生效，自定义所有视图的目标采样点数（可选）
#         """
#         self.long_transform = long_transform
#         self.short_transform = short_transform
#         self.num_short = num_short
#         self.long_crop_ratio = long_crop_ratio
#         self.short_crop_ratio = short_crop_ratio
#         self.sr = sr
#         self.target_length_mode = target_length_mode
#         fixed_length = 3000
#
#         # -------------------------- 新增：处理固定长度逻辑 --------------------------
#         if self.target_length_mode == "fixed":
#             # 若为固定长度模式，需传入fixed_length参数
#             assert "fixed_length" in locals(), "target_length_mode='fixed'时需传入fixed_length参数"
#             self.fixed_length = fixed_length
#             assert self.fixed_length > 0, "fixed_length必须为正整数"
#         elif self.target_length_mode != "short_crop":
#             raise ValueError("target_length_mode仅支持'short_crop'或'fixed'")
#
#         # 确保比例在合理范围内
#         assert 0 < self.long_crop_ratio <= 1, "长裁剪比例必须在(0, 1]之间"
#         assert 0 < self.short_crop_ratio <= self.long_crop_ratio, "短裁剪比例必须小于长裁剪比例且大于0"
#
#     def __call__(self, x):
#         """
#         对输入的EEG片段进行裁剪和增强（适配SupConLoss：统一所有视图长度）
#
#         参数:
#             x: 输入的EEG片段，形状通常为 [通道数, 采样点数] 或 [采样点数]
#
#         返回:
#             裁剪并增强后的片段列表（所有视图长度统一），第一个为长裁剪，其余为短裁剪
#         """
#         # 确保输入是numpy数组
#         if not isinstance(x, np.ndarray):
#             x = np.array(x)
#
#         # 获取输入的长度（采样点数）
#         if x.ndim == 1:
#             length = x.shape[0]  # 单通道情况
#         else:
#             length = x.shape[1]  # 多通道情况（通道数, 采样点数）
#
#         crops = []
#
#         # -------------------------- 新增：计算所有视图的目标统一长度 --------------------------
#         if self.target_length_mode == "short_crop":
#             # 按短裁剪长度统一（避免长裁剪压缩过多，也可改为按长裁剪长度，需调整逻辑）
#             target_length = int(length * self.short_crop_ratio)
#         else:  # fixed模式
#             target_length = self.fixed_length
#         # 确保目标长度合法（不超过原始长度，且至少1个采样点）
#         target_length = max(1, min(target_length, length))
#
#
#         # 1. 长裁剪（裁剪后需插值到目标长度）
#         # 计算长裁剪的原始长度（采样点数）
#         long_crop_length = int(length * self.long_crop_ratio)
#         long_crop_length = max(1, long_crop_length)
#         # 随机起始位置
#         long_start = random.randint(0, length - long_crop_length)
#         # 执行裁剪
#         if x.ndim == 1:
#             long_crop = x[long_start:long_start + long_crop_length]
#         else:
#             long_crop = x[:, long_start:long_start + long_crop_length]
#
#         # -------------------------- 新增：长裁剪插值到目标长度（关键：统一维度） --------------------------
#         if long_crop_length != target_length:
#             if x.ndim == 1:
#                 # 单通道：1D插值（线性插值，保留信号趋势）
#                 interp_func = interpolate.interp1d(
#                     x=np.arange(long_crop_length),  # 原始坐标
#                     y=long_crop,                    # 原始信号
#                     kind="linear",                  # 插值方式
#                     fill_value="extrapolate"        # 边缘填充（避免越界）
#                 )
#                 # 生成目标长度的坐标，插值得到统一长度的信号
#                 long_crop_unified = interp_func(np.linspace(0, long_crop_length - 1, target_length))
#             else:
#                 # 多通道：逐通道插值（保证每个通道的时间维度一致）
#                 long_crop_unified = []
#                 for ch in range(x.shape[0]):
#                     interp_func = interpolate.interp1d(
#                         x=np.arange(long_crop_length),
#                         y=long_crop[ch],
#                         kind="linear",
#                         fill_value="extrapolate"
#                     )
#                     long_crop_unified_ch = interp_func(np.linspace(0, long_crop_length - 1, target_length))
#                     long_crop_unified.append(long_crop_unified_ch)
#                 # 恢复多通道形状：[通道数, 目标长度]
#                 long_crop_unified = np.stack(long_crop_unified, axis=0)
#         else:
#             # 若长裁剪长度已等于目标长度，直接复用
#             long_crop_unified = long_crop
#
#         # 应用增强并添加到列表（此时长裁剪已与短裁剪长度统一）
#         crops.append(self.long_transform(long_crop_unified))
#
#
#         # 2. 短裁剪（多个，裁剪后长度已与目标长度一致）
#         # 计算短裁剪的长度（采样点数），直接等于目标长度
#         short_crop_length = target_length
#         # 确保短裁剪长度合法（短于长裁剪原始长度，避免逻辑矛盾）
#         short_crop_length = max(1, min(short_crop_length, long_crop_length - 1))
#
#         for _ in range(self.num_short):
#             # 随机起始位置（确保不越界）
#             short_start = random.randint(0, length - short_crop_length)
#             # 执行裁剪（长度已为目标长度，无需后续插值）
#             if x.ndim == 1:
#                 short_crop = x[short_start:short_start + short_crop_length]
#             else:
#                 short_crop = x[:, short_start:short_start + short_crop_length]
#             # 应用增强并添加到列表
#             crops.append(self.short_transform(short_crop))
#
#         # -------------------------- 输出说明：所有视图长度统一为target_length --------------------------
#         # 例如：原始3000采样点（30秒），target_length=750（7.5秒），则所有视图均为[1,750]（单通道）
#         return crops
###############################################################################################



class MultiCropTransform:
    def __init__(self, long_transform, short_transform, num_short=4,
                 long_crop_ratio=0.8, middle_crop_ratio=0.5, short_crop_ratio=0.25, sr=100):
        """
        多裁剪变换类（取消长度强制统一）

        参数:
            long_transform: 长裁剪的增强变换
            short_transform: 短裁剪的增强变换
            num_short: 短裁剪的数量
            long_crop_ratio: 长裁剪占原始长度的比例（0-1之间）
            short_crop_ratio: 短裁剪占原始长度的比例（0-1之间）
            sr: 采样率，用于计算实际时长
        """
        self.long_transform = long_transform
        self.short_transform = short_transform
        self.num_short = num_short
        self.long_crop_ratio = long_crop_ratio
        self.middle_crop_ratio = middle_crop_ratio
        self.short_crop_ratio = short_crop_ratio
        self.sr = sr

        # 确保比例在合理范围内
        assert 0 < self.long_crop_ratio <= 1, "长裁剪比例必须在(0, 1]之间"
        assert 0 < self.short_crop_ratio <= self.long_crop_ratio, "短裁剪比例必须小于长裁剪比例且大于0"

    def __call__(self, x):
        """
        对输入的EEG片段进行裁剪和增强（不统一长度，保留原始裁剪尺寸）

        参数:
            x: 输入的EEG片段，形状通常为 [通道数, 采样点数] 或 [采样点数]

        返回:
            裁剪并增强后的片段列表（长裁剪和短裁剪长度不同）
        """
        # 确保输入是numpy数组
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # 获取输入的长度（采样点数）
        if x.ndim == 1:
            length = x.shape[0]  # 单通道情况
        else:
            length = x.shape[1]  # 多通道情况（通道数, 采样点数）

        crops = []

        # 1. 长裁剪（保留原始长度，不插值）
        # 计算长裁剪的原始长度（采样点数）
        long_crop_length = int(length * self.long_crop_ratio)
        long_crop_length = max(1, long_crop_length)
        # 随机起始位置
        long_start = random.randint(0, length - long_crop_length)
        # 执行裁剪（不做长度统一）
        if x.ndim == 1:
            long_crop = x[long_start:long_start + long_crop_length]
        else:
            long_crop = x[:, long_start:long_start + long_crop_length]
        # 应用增强并添加到列表
        crops.append(self.long_transform(long_crop))

        # 2. 短裁剪（保留原始长度，不插值）
        # 计算短裁剪的长度（采样点数）
        short_crop_length = int(length * self.short_crop_ratio)
        short_crop_length = max(1, short_crop_length)
        # 确保短裁剪长度不超过原始长度
        short_crop_length = min(short_crop_length, length)

        for _ in range(self.num_short):
            # 随机起始位置（确保不越界）
            short_start = random.randint(0, length - short_crop_length)
            # 执行裁剪（不做长度统一）
            if x.ndim == 1:
                short_crop = x[short_start:short_start + short_crop_length]
            else:
                short_crop = x[:, short_start:short_start + short_crop_length]
            # 应用增强并添加到列表
            crops.append(self.short_transform(short_crop))

        # 输出：长裁剪和短裁剪长度不同（由各自的比例决定）
        return crops



class Compose:

    def __init__(self, transforms, mode='full'):
        self.transforms = transforms
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'random':
            index = random.randint(0, len(self.transforms) - 1)
            x = self.transforms[index](x)
        elif self.mode == 'full':
            for t in self.transforms:
                x = t(x)
        elif self.mode == 'shuffle':
            transforms = np.random.choice(self.transforms, len(self.transforms), replace=False)
            for t in transforms:
                x = t(x)
        else:
            raise NotImplementedError
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomAmplitudeScale:

    def __init__(self, range=(0.5, 2.0), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            scale = random.uniform(self.range[0], self.range[1])
            return x * scale
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomDCShift:
    
    def __init__(self, range=(-10.0, 10.0), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            shift = random.uniform(self.range[0], self.range[1])
            return x + shift
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTimeShift:

    def __init__(self, range=(-300, 300), mode='constant', cval=0.0, p=0.5):
        self.range = range
        self.mode = mode
        self.cval = cval
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            t_shift = random.randint(self.range[0], self.range[1])
            if len(x.shape) == 2:
                x = x[0]
            x = shift(input=x, shift=t_shift, mode=self.mode, cval=self.cval)
            x = np.expand_dims(x, axis=0)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


# class RandomZeroMasking:
#
#     def __init__(self, range=(0, 300), p=0.5):
#         self.range = range
#         self.p = p
#
#     def __call__(self, x):
#         if torch.rand(1) < self.p:
#             mask_len = random.randint(self.range[0], self.range[1])
#             random_pos = random.randint(0, x.shape[1] - mask_len)
#             # print("DEBUG: x.shape=", x.shape, "mask_len=", mask_len)
#
#             mask = np.concatenate([np.ones((1, random_pos)), np.zeros((1, mask_len)), np.ones((1, x.shape[1] - mask_len - random_pos))], axis=1)
#             return x * mask
#         return x
#
#     def __repr__(self):
#         return self.__class__.__name__ + '()'


class RandomZeroMasking:

    def __init__(self, mask_range=(0, 300), p=0.5):
        self.mask_range = mask_range
        self.p = p

    def __call__(self, x):
        # 1. 形状标准化 (Shape Normalization)
        # 你的数据可能是 numpy array 也可能是 tensor，先统一处理维度
        # 目标是将 (1, 1, 1500, 1) 这种多维数据变成 (C, Time)

        input_is_tensor = isinstance(x, torch.Tensor)
        if not input_is_tensor:
            x = torch.from_numpy(x)  # 临时转为 tensor 方便处理形状

        # 如果维度过多 (>2)，说明有很多无用的维度 1
        if x.ndim > 2:
            # squeeze() 会去掉所有为 1 的维度，比如 (1, 1, 1500, 1) -> (1500,)
            x = x.squeeze()
            # 如果挤压过头变成了 1维 (1500,)，再加回一个 Channel 维度 -> (1, 1500)
            if x.ndim == 1:
                x = x.unsqueeze(0)

        # 转回 numpy (如果你后面的代码需要 numpy)
        # 或者如果你后面的代码全是 tensor，这里就不用转回去
        # 这里的实现假设我们要保持和原来一样的 numpy 运算逻辑
        if not input_is_tensor and isinstance(x, torch.Tensor):
            x = x.numpy()

        # -----------------------------------------------------------
        # 下面是正常的 Mask 逻辑 (此时 x.shape 应该是 [1, 1500] 左右)
        # -----------------------------------------------------------
        seq_len = x.shape[1]

        # 再次检查：只打印那些真正奇怪的长度 (比如 Multi-Crop 产生的碎片)
        # 标准的 3000 和 1500 不打印
        if seq_len != 3000 and seq_len != 1500:
            # 这里可以保留你的 debug，或者注释掉
            pass

        if torch.rand(1) < self.p:
            # 安全检查
            if seq_len <= self.mask_range[0]:
                return x

            # 动态调整上限
            max_mask_len = min(self.mask_range[1], seq_len)

            # 随机 mask 长度
            mask_len = random.randint(self.mask_range[0], max_mask_len)

            # 随机位置
            random_pos = random.randint(0, seq_len - mask_len)

            # 构建 mask
            mask = np.concatenate([
                np.ones((1, random_pos)),
                np.zeros((1, mask_len)),
                np.ones((1, seq_len - mask_len - random_pos))
            ], axis=1)

            # 确保类型匹配 (如果是 Tensor，转 numpy mask 为 tensor)
            if isinstance(x, torch.Tensor):
                mask = torch.from_numpy(mask).to(x.device).type_as(x)

            return x * mask

        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomAdditiveGaussianNoise:
    
    def __init__(self, range=(0.0, 0.2), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            sigma = random.uniform(self.range[0], self.range[1])
            return x + np.random.normal(0, sigma, x.shape)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomBandStopFilter:

    def __init__(self, range=(0.5, 30.0), band_width=2.0, sampling_rate=100.0, p=0.5):
        self.range = range
        self.band_width = band_width
        self.sampling_rate = sampling_rate
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            low_freq = random.uniform(self.range[0], self.range[1])
            center_freq = low_freq + self.band_width / 2.0
            b, a = signal.iirnotch(center_freq, center_freq / self.band_width, fs=self.sampling_rate)
            x = signal.lfilter(b, a, x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
