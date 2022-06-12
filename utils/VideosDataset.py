import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from random import random
from PIL import Image


# +
#returns sequences for images for CNN-LSTM training
class VideosDataset(Dataset):
    SEQ_LEN=598
    def __init__(self, transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        #read all images paths,label dic
        with open('./pickle/vids2label.p', 'rb') as fp:
            vids2label = pickle.load(fp)
            
            
        self.ROOT='./e6691-bucket-images/'
        self.vids2label=vids2label
        self.n_samples = len(self.vids2label)
        self.index2data=[]
        self.convert_tensor = transforms.Compose([transforms.PILToTensor()])
        for vid_name in self.vids2label:
            self.index2data.append(vid_name)
            
        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        path=os.path.join(self.ROOT,self.index2data[index])
        labels=self.vids2label[self.index2data[index]]
        X=[]
        rand=min(int(random()*(len(labels)-VideosDataset.SEQ_LEN)),len(labels)-VideosDataset.SEQ_LEN)
        
        #for i in range(len(labels)):
        
        for i in range(rand,rand+VideosDataset.SEQ_LEN):
            img_path=os.path.join(path,'frame'+str(i)+'.jpg')            
            X.append(torch.Tensor.float(self.convert_tensor(Image.open(img_path))))
        X=torch.stack(X)
        if self.transform:
            self.transform(X),torch.tensor(labels)
        #return X, torch.tensor(labels)
        return X, torch.tensor(labels[rand:rand+VideosDataset.SEQ_LEN])

        #return torch.Tensor.float(self.convert_tensor(Image.open(path))), torch.nn.functional.one_hot(torch.tensor(self.vids2label[self.index2data[index]],dtype=torch.long), num_classes=17)
   
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    


