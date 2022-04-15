import torch.nn as nn
from .alexnet import *
from .convnext import *
from .densenet import *
from .efficientnet import *
from .googlenet import *
from .inception import *
from .mnasnet import *
from .mobilenet import *
from .regnet import *
from .resnet import *
from .shufflenetv2 import *
from .squeezenet import *
from .vgg import *
from .vision_transformer import *
from . import detection
from . import optical_flow
from . import quantization
from . import segmentation
from . import video
from ._api import get_weight

class BaseModel(nn.Module):
    def __init__(self, name, num_classes):
        super(BaseModel, self).__init__()
        if name == 'alexnet':
            self.base = alexnet(num_classes=num_classes)
        elif name == 'vggn16':
            self.base = vgg16(num_classes=num_classes)
        elif name == 'googlenet':
            self.base = GoogLeNet(num_classes=num_classes)
        elif name == 'resnet18':
            self.base = resnet18(num_classes=num_classes)    # pretrained=True 是加载预训练，
        elif name == 'resnet34':
            self.base = resnet34(num_classes=num_classes)
        elif name == 'resnet50':
            self.base = resnet50(num_classes=num_classes)
        elif name == 'densenet121':
            self.base = densenet121(num_classes=num_classes)
        elif name == 'mobilenetv2':
            self.base = MobileNetV2(num_classes=num_classes)
        elif name == 'mobilenetv3':
            self.base = MobileNetV3(num_classes=num_classes)
        elif name == 'shufflenetv2':
            self.base = ShuffleNetV2(num_classes=num_classes)
        elif name == 'SqueezeNet1_0_Weights':
            self.base = SqueezeNet1_0_Weights(num_classes=num_classes)
        else:  #还有更多自己添加
            raise ValueError('Input model name is not supported!!!')

    def forward(self, x):
        return self.base(x)

