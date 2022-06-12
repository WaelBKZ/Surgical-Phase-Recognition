# +
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

from PIL import Image


# -

# Range: RALIHR_surgeon01_fps01_0071 - RALIHR_surgeon01_fps01_0125
#
# Range: RALIHR_surgeon02_fps01_0001 - RALIHR_surgeon02_fps01_0004
#
# RALIHR_surgeon03_fps01_0001
#

# experimental, please ignore
class TestImagesDataset(Dataset):
    
    def __init__(self, transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        self.paths=[]
        for i in range(71,126):
            self.paths.append('RALIHR_surgeon01_fps01_'+('0000'+str(i))[-4:])

        for i in range(1,5):
            self.paths.append('RALIHR_surgeon02_fps01_'+('0000'+str(i))[-4:])

        self.paths.append('RALIHR_surgeon03_fps01_0001')

                
        self.ROOT='./e6691-bucket-images/'
        
        self.index2data=[]
        self.convert_tensor = transforms.Compose([transforms.PILToTensor()])
        
        
        for vid in self.paths:
            for i in range (len(os.listdir(os.path.join(self.ROOT,vid)))):
                self.index2data.append(os.path.join(vid,'frame'+str(i)+'.jpg'))
        self.transform = transform
        self.n_samples = len(self.index2data)
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        path=os.path.join(self.ROOT,self.index2data[index])
        if self.transform:
            self.transform(torch.Tensor.float(self.convert_tensor(Image.open(path))))
        return torch.Tensor.float(self.convert_tensor(Image.open(path)))

        #return torch.Tensor.float(self.convert_tensor(Image.open(path))), torch.nn.functional.one_hot(torch.tensor(self.frames2label[self.index2data[index]],dtype=torch.long), num_classes=17)
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


