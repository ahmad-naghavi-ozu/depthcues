import torch.nn as nn
import timm
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode, ToTensor


class ResNetBackbone(nn.Module):
    def __init__(self, model_name="resnet50.tv_in1k"):
        '''
        ResNet 18: feat_dim = [64,128,256,512], layer = [1,2,3,4]
        ResNet 50: feat_dim = [256,512,1024,2048], layer = [1,2,3,4]
        '''
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=0)
        self.feat_dim = {
            "resnet18.tv_in1k": [None,64,128,256,512],
            "resnet50.tv_in1k": [None,256,512,1024,2048],
            "resnext50_32x4d.tv_in1k": [None,256,512,1024,2048],
            "senet154.gluon_in1k": [None,256,512,1024,2048]
        }[model_name]

    def forward_intermediates(self, images, layer):
        out = self.model.forward_intermediates(images, indices=[layer], stop_early=True, intermediates_only=True)
        out = out[0]

        return [(out, None)]
    
def get_resnet_transform():
    return Compose(
        [
            Resize(size=(518,518), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
        ]
    )