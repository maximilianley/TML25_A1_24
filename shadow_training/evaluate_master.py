import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
from random import randint
from torchvision.models import resnet18
import __main__

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # no need during evaluation

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]: # __getitem__ is for instance called implicitly by an iterator
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

torch.serialization.add_safe_globals([__main__.MembershipDataset])
torch.serialization.add_safe_globals([__main__.TaskDataset])

class MetaClassifier(nn.Module):
    def __init__(self):
        super(MetaClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(44, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
        

print("Load private dataset")
data_priv: MembershipDataset = torch.load("../priv_out.pt")

print("Load master model.")
model_master = MetaClassifier()
ckpt = torch.load("./master_linear_44_64_32_2.pt", map_location="cpu")
model_master.load_state_dict(ckpt)

print("Load target model.")
# recreate the same architecture (very important)
model_target = resnet18(pretrained=False)
model_target.fc = torch.nn.Linear(512, 44)
#ckpt = torch.load("./test_model_weights.pt", map_location="cpu")
ckpt = torch.load("../01_MIA.pt", map_location="cpu")
model_target.load_state_dict(ckpt)

print("Models loaded successfully.")


print("Evaluate")
model_target.eval()
model_master.eval()
test_index = 539
with torch.no_grad():
    ok_sample = data_priv[test_index][1].unsqueeze(0)
    output = model_target(ok_sample)
    probs = torch.nn.functional.softmax(output, dim=1)
    print(str(probs))
    
    output = model_master(probs)
    output = torch.nn.functional.softmax(output, dim=1)
    print(str(output))
    print(str(float(output[0][1])))
    print(str(data_priv[test_index][3]))
    
    
print("check accuracy")
print("Load public dataset")
shadow_pub: TaskDataset = torch.load("./evaluated_shadow_pub_shuffled.pt")
test_amt = 40000
mid = 0.5
misses = 0
with torch.no_grad():
    for j in range(test_amt):
        idx = randint(0, 19999)
        ok_sample = shadow_pub[idx][1].unsqueeze(0)
        ok_membership = int(shadow_pub[idx][2])
        output = model_master(ok_sample)
        #print(str(output))
        probs = torch.nn.functional.softmax(output[0], dim=1)
        #print(str(probs))
        #print(str(probs[0][0]))
        #print(str(float(probs[0][0][0])))
        mem_val = float(probs[0][0])
        non_val = float(probs[0][1])
        if mem_val > mid:
            mem_val = 1
        else:
            mem_val = 0
        mult = (mem_val * ok_membership)**2
        misses += mult
        
print(f"Accuracy: {(test_amt - misses)/(test_amt)}")
