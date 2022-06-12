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

# Images dataset for CNN training
class ImagesDataset(Dataset):
    
    def __init__(self, transform=None,train=True,noval=False):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        #read all images paths,label dic
        with open('./pickle/frames2label.p', 'rb') as fp:
            frames2label = pickle.load(fp)
        self.train_dic={}
        self.noval=noval
        self.val_dic={}
        # validation videos
        self.l=[45, 33, 8, 16, 60, 28, 30, 5, 67, 25, 46, 52, 69, 17, 47, 26, 24, 18, 66, 19]# validation set list
        if self.noval:
            self.l=[]
        for k in frames2label:
            if not int(k[23:k.index('/')]) in self.l:
                self.train_dic[k]=frames2label[k]
            else:
                self.val_dic[k]=frames2label[k]
                
                
        self.ROOT='./e6691-bucket-images/'
        self.frames2label=self.train_dic
        if not train:
            self.frames2label=self.val_dic
        self.n_samples = len(self.frames2label)
        self.index2data=[]
        self.convert_tensor = transforms.Compose([transforms.PILToTensor()])
        
        
        for frame in self.frames2label:
            self.index2data.append(frame)
        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        path=os.path.join(self.ROOT,self.index2data[index])
        if self.transform:
            self.transform(torch.Tensor.float(self.convert_tensor(Image.open(path)))),torch.tensor(self.frames2label[self.index2data[index]])
        return torch.Tensor.float(self.convert_tensor(Image.open(path))), torch.tensor(self.frames2label[self.index2data[index]])

        #return torch.Tensor.float(self.convert_tensor(Image.open(path))), torch.nn.functional.one_hot(torch.tensor(self.frames2label[self.index2data[index]],dtype=torch.long), num_classes=17)
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
