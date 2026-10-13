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

class MultiCropTransform:
    def __init__(self, long_transform, short_transform, num_short=4,
                 long_crop_ratio=0.8, middle_crop_ratio=0.5, short_crop_ratio=0.25, sr=100):
        """
        Multi-crop transform that preserves each crop's original length.

        Args:
            long_transform: Augmentation applied to long crops.
            short_transform: Augmentation applied to short crops.
            num_short: Number of short crops.
            long_crop_ratio: Long-crop ratio relative to the original length.
            short_crop_ratio: Short-crop ratio relative to the original length.
            sr: Sampling rate used to calculate the actual duration.
        """
        self.long_transform = long_transform
        self.short_transform = short_transform
        self.num_short = num_short
        self.long_crop_ratio = long_crop_ratio
        self.middle_crop_ratio = middle_crop_ratio
        self.short_crop_ratio = short_crop_ratio
        self.sr = sr

        # Ensure crop ratios are within valid ranges
        assert 0 < self.long_crop_ratio <= 1, "长裁剪比例必须在(0, 1]之间"
        assert 0 < self.short_crop_ratio <= self.long_crop_ratio, "短裁剪比例必须小于长裁剪比例且大于0"

    def __call__(self, x):
        """
        Crop and augment an EEG segment while preserving crop lengths.

        Args:
            x: EEG segment shaped [channels, samples] or [samples].

        Returns:
            Augmented crops with different long- and short-crop lengths.
        """
        # Ensure the input is a NumPy array
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # Get the input length in samples
        if x.ndim == 1:
            length = x.shape[0]  # Single-channel input
        else:
            length = x.shape[1]  # Multi-channel input: [channels, samples]

        crops = []

        # 1. Long crop without interpolation
        # Calculate the long-crop length in samples
        long_crop_length = int(length * self.long_crop_ratio)
        long_crop_length = max(1, long_crop_length)
        # Select a random start position
        long_start = random.randint(0, length - long_crop_length)
        # Extract the crop without normalizing its length
        if x.ndim == 1:
            long_crop = x[long_start:long_start + long_crop_length]
        else:
            long_crop = x[:, long_start:long_start + long_crop_length]
        # Apply augmentation and append the crop
        crops.append(self.long_transform(long_crop))

        # 2. Short crops without interpolation
        # Calculate the short-crop length in samples
        short_crop_length = int(length * self.short_crop_ratio)
        short_crop_length = max(1, short_crop_length)
        # Ensure the short crop does not exceed the input length
        short_crop_length = min(short_crop_length, length)

        for _ in range(self.num_short):
            # Select a random valid start position
            short_start = random.randint(0, length - short_crop_length)
            # Extract the crop without normalizing its length
            if x.ndim == 1:
                short_crop = x[short_start:short_start + short_crop_length]
            else:
                short_crop = x[:, short_start:short_start + short_crop_length]
            # Apply augmentation and append the crop
            crops.append(self.short_transform(short_crop))

        # Long and short outputs retain lengths determined by their ratios
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

class RandomZeroMasking:

    def __init__(self, mask_range=(0, 300), p=0.5):
        self.mask_range = mask_range
        self.p = p

    def __call__(self, x):
        # 1. Normalize the input shape
        # Handle NumPy arrays and tensors consistently
        # Convert shapes such as (1, 1, 1500, 1) to (C, Time)

        input_is_tensor = isinstance(x, torch.Tensor)
        if not input_is_tensor:
            x = torch.from_numpy(x)  # Temporarily convert to a tensor for reshaping

        # Extra dimensions indicate redundant singleton axes
        if x.ndim > 2:
            # squeeze() removes singleton axes: (1, 1, 1500, 1) -> (1500,)
            x = x.squeeze()
            # Restore a channel axis if squeezing produces a 1D signal
            if x.ndim == 1:
                x = x.unsqueeze(0)

        # Convert back to NumPy to preserve the original downstream behavior
        if not input_is_tensor and isinstance(x, torch.Tensor):
            x = x.numpy()

        # -----------------------------------------------------------
        # Apply masking after x has been normalized to roughly [1, 1500]
        # -----------------------------------------------------------
        seq_len = x.shape[1]

        # Identify unusual lengths, such as fragments produced by MultiCrop
        # Standard lengths of 3000 and 1500 need no debug output
        if seq_len != 3000 and seq_len != 1500:
            # Add optional debug output here if needed
            pass

        if torch.rand(1) < self.p:
            # Guard against signals shorter than the minimum mask length
            if seq_len <= self.mask_range[0]:
                return x

            # Adjust the upper bound to the sequence length
            max_mask_len = min(self.mask_range[1], seq_len)

            # Select a random mask length
            mask_len = random.randint(self.mask_range[0], max_mask_len)

            # Select a random mask position
            random_pos = random.randint(0, seq_len - mask_len)

            # Build the mask
            mask = np.concatenate([
                np.ones((1, random_pos)),
                np.zeros((1, mask_len)),
                np.ones((1, seq_len - mask_len - random_pos))
            ], axis=1)

            # Match the mask type and device when x is a tensor
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
