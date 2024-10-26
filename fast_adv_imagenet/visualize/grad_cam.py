import sys
sys.path.append("..")
import numpy as np

from fast_adv_imagenet.visualize.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):  # (1, 256, 56, 56)
        return np.mean(grads, axis=(2, 3))
