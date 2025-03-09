import torch
from utils import instantiate_from_config


class MultiCueLoss:
    def __init__(
            self,
            loss_config=None,
            cues=['occlusion', 'lightshadow', 'perspective', 'size', 'texturegrad', 'elevation'], 
    ):
        super().__init__()

        self.cues = cues

        self.cue_indices = {}
        left, right = 0, 1 # exclusive
        for cue in cues:
            if cue in ['perspective', 'elevation']:
                right += 1
            self.cue_indices[cue] = (left, right)
            left = right
            right += 1

        self.loss_functions = {
            cue: instantiate_from_config(loss_config[cue]) for cue in cues
        }

        self.loss_weights = {
            cue: loss_config[cue].get('weight', 1.) for cue in cues
        }