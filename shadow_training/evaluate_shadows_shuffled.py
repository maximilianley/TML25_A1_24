import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
from random import randint
from torchvision.models import resnet18
import __main__
from random import randint

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

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
data_pub: MembershipDataset = torch.load("../pub.pt")
print(type(data_pub))
print(str(data_pub))


########### split dataset ##################


print(f"Divide dataset into members and non-members.")

amt_models = 10
int_is_member = 1
int_is_not_member = 0
data_amt = len(data_pub)

class My_Sample():
    def __init__(self):
        img = None
        _id = None
        label = None
        is_member = None
        
    def sample_is_member(self):
        return self.is_member == int_is_member

pub_mem = []
pub_non_mem = []

for s in data_pub:

    s_wrapper = My_Sample()
    s_wrapper.img = s[1]
    s_wrapper._id = s[0]
    s_wrapper.label = s[2]
    s_wrapper.is_member = s[3]
    
    if s[3] == int_is_member:
        pub_mem.append(s_wrapper)
    elif s[3] == int_is_not_member:
        pub_non_mem.append(s_wrapper)
    else:
        print("This should not happen.\nSample should either be a member or non-member")
        exit()
        
if len(pub_mem) != len(pub_non_mem):
    print("It's expected that pub_mem and pub_non_mem have the same amount of samples")
    exit()
    
    
print(f"Partition dataset into {amt_models} parts.")

shadow_partitions = []
for i in range(amt_models):
    shadow_partitions.append([])

print("Members")
counter = 0
shadow_set_counter = 0
shadow_amt = int(len(pub_mem)/amt_models) # floor down
shadow_set = None
for w in pub_mem:
    if counter == 0:
        shadow_set = []
    shadow_set.append(w)
    counter += 1
    if counter == shadow_amt:
        shadow_partitions[shadow_set_counter].append(shadow_set)
        counter = 0 # reset
        shadow_set_counter += 1
        if shadow_set_counter == amt_models:
            break
            
print("Non-Members")
counter = 0
shadow_set_counter = 0
shadow_amt = int(len(pub_mem)/amt_models) # floor down
shadow_set = None
for w in pub_non_mem:
    if counter == 0:
        shadow_set = []
    shadow_set.append(w)
    counter += 1
    if counter == shadow_amt:
        shadow_partitions[shadow_set_counter].append(shadow_set)
        counter = 0 # reset
        shadow_set_counter += 1
        if shadow_set_counter == amt_models:
            break
            
last_entry = shadow_partitions[len(shadow_partitions) - 1]
if len(last_entry[0]) != len(last_entry[1]):
    print("Amount of member and non-member subsets should be the same.")
    exit()



############################################


total = 10
conf_dataset = TaskDataset()
for i in range(total):
    idx = i + 1
    print(f"Shadow Model {idx}.")
    
    ckpt = torch.load(f"./shadow_models/shadow_{idx}.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    
    with torch.no_grad():
        mem_data = shadow_partitions[i][0]
        non_mem_data = shadow_partitions[i][1]
        run_mem = 0
        run_non = 0
        l_mem = len(mem_data)
        l_non = len(non_mem_data)
        
        while(run_mem < l_mem or run_non < l_non):
            r = randint(0, 1)
            w = None
            if r == 0:
                if run_mem >= l_mem:
                    if run_non >= l_non:
                        print("Shouldnt happen.")
                        exit()
                    else:
                        w = non_mem_data[run_non]
                        run_non += 1
                else:
                    w = mem_data[run_mem]
                    run_mem += 1
            else:
                if run_non >= l_non:
                    if run_mem >= l_mem:
                        print("Shouldnt happen.")
                        exit()
                    else:
                        w = mem_data[run_mem]
                        run_mem += 1
                else:
                    w = non_mem_data[run_non]
                    run_non += 1
                    
            ok_sample = w.img.unsqueeze(0)
            output = model(ok_sample)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            conf_dataset.imgs.append(1*probs) # yeah, problem?
            #conf_dataset.imgs.append(output)
            conf_dataset.ids.append(w._id)
            conf_dataset.labels.append(w.is_member)
            
        # I should normalize inputs to [0;1] conf. ranges first
        
torch.save(conf_dataset, 'evaluated_shadow_pub_shuffled.pt')
print("Fädisch")
        
