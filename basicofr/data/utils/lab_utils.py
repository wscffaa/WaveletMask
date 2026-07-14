import torch
from typing import Sequence, Union
import numpy as np


def _is_tensor_image(img: torch.Tensor) -> bool:
    return torch.is_tensor(img) and img.ndimension() == 3


def _normalize(tensor: torch.Tensor,
               mean: Union[float, Sequence[float]],
               std: Union[float, Sequence[float]]) -> torch.Tensor:
    if not _is_tensor_image(tensor):
        raise TypeError('Expected tensor image with 3 dimensions.')
    if tensor.size(0) == 1:
        tensor.sub_(mean).div_(std)
    else:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
    return tensor


def Normalize_LAB(inputs: torch.Tensor) -> torch.Tensor:
    inputs = inputs.clone()
    inputs[0:1, :, :] = _normalize(inputs[0:1, :, :], 50.0, 1.0)
    inputs[1:3, :, :] = _normalize(inputs[1:3, :, :], (0.0, 0.0), (1.0, 1.0))
    return inputs


def to_mytensor(pic) -> torch.Tensor:
    if isinstance(pic, torch.Tensor):
        tensor = pic
    else:
        arr = np.array(pic)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        tensor = torch.from_numpy(arr.transpose((2, 0, 1)))
    return tensor.float()
