# +
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

import random


# -

#returns sequences for images for LSTM error correction training
class SequenceDataset(Dataset):
    
    def __init__(self, transform=None,train=True):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        #read all images paths,label dic
        self.train=train
        with open('./pickle/vids2label.p', 'rb') as fp:
            vids2label = pickle.load(fp)
        self.vids2label=vids2label
        self.train_dic={}
        self.val_dic={}
        self.l=[45, 33, 8, 16, 60, 28, 30, 5, 67, 25, 46, 52, 69, 17, 47, 26, 24, 18, 66, 19]# validation set list
        for k in self.vids2label:
            if not int(k[23:]) in self.l:
                self.train_dic[k]=self.vids2label[k]
            else:
                self.val_dic[k]=self.vids2label[k]
                
                
        self.ROOT='./e6691-bucket-images/'
        self.vids2label=self.train_dic
        if not train:
            self.vids2label=self.val_dic
        self.n_samples = len(self.vids2label)
        self.index2data=[]
        self.convert_tensor = transforms.Compose([transforms.PILToTensor()])
        
        
        for vid in self.vids2label:
            self.index2data.append(vid)
        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        
        data=self.vids2label[self.index2data[index]]
        l=len(data)
        s=0
        if self.train:
            l=len(data)//2
            s=random.randint(0,l-1)
            
        level=random.random()*0.3
        corrupted_data=[label if random.random()>level else random.randint(0, 13) for label in data]
        
        data=np.asarray(data[s:s+l])
        corrupted_data=np.asarray(corrupted_data[s:s+l])
    
        
        b = np.zeros((corrupted_data.size, 14)) 
        b[np.arange(corrupted_data.size),corrupted_data] = 1
        if self.transform:
            self.transform(torch.Tensor.float(torch.from_numpy(b))),torch.Tensor.float(torch.from_numpy(data))
        return torch.Tensor.float(torch.from_numpy(b)),torch.Tensor.float(torch.from_numpy(data))

        #return torch.Tensor.float(self.convert_tensor(Image.open(path))), torch.nn.functional.one_hot(torch.tensor(self.frames2label[self.index2data[index]],dtype=torch.long), num_classes=17)
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
