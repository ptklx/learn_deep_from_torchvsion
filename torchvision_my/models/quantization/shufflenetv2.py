from functools import partial
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision_my.models import shufflenetv2

from ...transforms._presets import ImageClassification, InterpolationMode
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param
from ..shufflenetv2 import ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights
from .utils import _fuse_modules, _replace_relu, quantize_model


__all__ = [
    "QuantizableShuffleNetV2",
    "ShuffleNet_V2_X0_5_QuantizedWeights",
    "ShuffleNet_V2_X1_0_QuantizedWeights",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
]


class QuantizableInvertedResidual(shufflenetv2.InvertedResidual):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = self.cat.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = self.cat.cat([self.branch1(x), self.branch2(x)], dim=1)

        out = shufflenetv2.channel_shuffle(out, 2)

        return out


class QuantizableShuffleNetV2(shufflenetv2.ShuffleNetV2):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, inverted_residual=QuantizableInvertedResidual, **kwargs)  # type: ignore[misc]
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        r"""Fuse conv/bn/relu modules in shufflenetv2 model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        for name, m in self._modules.items():
            if name in ["conv1", "conv5"] and m is not None:
                _fuse_modules(m, [["0", "1", "2"]], is_qat, inplace=True)
        for m in self.modules():
            if type(m) is QuantizableInvertedResidual:
                if len(m.branch1._modules.items()) > 0:
                    _fuse_modules(m.branch1, [["0", "1"], ["2", "3", "4"]], is_qat, inplace=True)
                _fuse_modules(
                    m.branch2,
                    [["0", "1", "2"], ["3", "4"], ["5", "6", "7"]],
                    is_qat,
                    inplace=True,
                )


def _shufflenetv2(
    stages_repeats: List[int],
    stages_out_channels: List[int],
    *,
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "fbgemm")

    model = QuantizableShuffleNetV2(stages_repeats, stages_out_channels, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "task": "image_classification",
    "architecture": "ShuffleNetV2",
    "publication_year": 2018,
    "size": (224, 224),
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "backend": "fbgemm",
    "quantization": "Post Training Quantization",
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
}


class ShuffleNet_V2_X0_5_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x0.5_fbgemm-00845098.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 1366792,
            "unquantized": ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
            "acc@1": 57.972,
            "acc@5": 79.780,
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1


class ShuffleNet_V2_X1_0_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-db332c57.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 2278604,
            "unquantized": ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
            "acc@1": 68.360,
            "acc@5": 87.582,
        },
    )
    DEFAULT = IMAGENET1K_FBGEMM_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: ShuffleNet_V2_X0_5_QuantizedWeights.IMAGENET1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
    )
)
def shufflenet_v2_x0_5(
    *,
    weights: Optional[Union[ShuffleNet_V2_X0_5_QuantizedWeights, ShuffleNet_V2_X0_5_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        weights (ShuffleNet_V2_X0_5_QuantizedWeights or ShuffleNet_V2_X0_5_Weights, optional): The pretrained
            weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    weights = (ShuffleNet_V2_X0_5_QuantizedWeights if quantize else ShuffleNet_V2_X0_5_Weights).verify(weights)
    return _shufflenetv2(
        [4, 8, 4], [24, 48, 96, 192, 1024], weights=weights, progress=progress, quantize=quantize, **kwargs
    )


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: ShuffleNet_V2_X1_0_QuantizedWeights.IMAGENET1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
    )
)
def shufflenet_v2_x1_0(
    *,
    weights: Optional[Union[ShuffleNet_V2_X1_0_QuantizedWeights, ShuffleNet_V2_X1_0_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        weights (ShuffleNet_V2_X1_0_QuantizedWeights or ShuffleNet_V2_X1_0_Weights, optional): The pretrained
            weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    weights = (ShuffleNet_V2_X1_0_QuantizedWeights if quantize else ShuffleNet_V2_X1_0_Weights).verify(weights)
    return _shufflenetv2(
        [4, 8, 4], [24, 116, 232, 464, 1024], weights=weights, progress=progress, quantize=quantize, **kwargs
    )
