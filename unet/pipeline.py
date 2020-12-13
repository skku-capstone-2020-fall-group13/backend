from typing import *
import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from .transform import Unfold

class UnetPipeline(object):
    def __init__(self):
        self.model = None
        self.device = None
        self.transform_img = None
    
    def load(self, model_dir: str, device: object):
        """
        model_dir: must have 'config.json' and 'pytorch_model.pt'
        device: torch.device object
        """
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        self.crop_size = config['crop_size']
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            Unfold(crop_size=self.crop_size, stride=self.crop_size),
        ])

        self.model = smp.Unet(**config['smp_args'])
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.pt'),map_location=device))

        self.device = device
        self.model.to(self.device)
        
    def predict(self, image: np.ndarray, batch_size=512):
        """
        image: Array[1, H, W, 3(RGB)]
        """
        assert self.model, "Model is not loaded"

        cropped_images = self.transform_img(image)
        cropped_set = TensorDataset(cropped_images)
        cropped_loader = DataLoader(cropped_set, batch_size=batch_size)

        self.model.eval()
        result = None
        with torch.no_grad():
            for images in cropped_loader:
                images = images[0].to(self.device)
                
                outputs = self.model(images)
                _, prediction = outputs.max(1)
                
                if result is None:
                    result = prediction
                else:
                    result = torch.cat([result, prediction], dim=0)

        H_cnt, W_cnt = image.shape[0]//self.crop_size, image.shape[1]//self.crop_size
        result = result.reshape(H_cnt, W_cnt, self.crop_size,self.crop_size)
        result = result.permute(0,2,1,3).reshape(*image.shape[:2])

        return result.cpu().numpy()

        



