import torch
from torch.utils.data import Dataset
from torch import optim
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import requests
import pandas as pd
from random import randint
from torchvision import transforms
import __main__
import model_constants as k


#### LOADING THE MODEL

from torchvision.models import resnet18


#### DATASETS

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
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

# __module__ = "__main__"
torch.serialization.add_safe_globals([__main__.MembershipDataset])
print()

# LOAD DATASETS
print("Load Public dataset.")
data_pub: MembershipDataset = torch.load("../pub.pt")

print("Load Private dataset.")
data_priv: MembershipDataset = torch.load("../priv_out.pt")

amt_models = 10
if len(data_pub) % amt_models != 0:
    exit()
print(f"Train {amt_models} shadow models.")

print(f"Divide dataset into members and non-members.")

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



#### MAIN TRAINING LOOP
print()
print("Set training parameters")

epochs = 20
loss_abort = 0.019 # 0.005
loss_delta_abort = 0.0011
loss_delta_list = [4, 2, 3, 1]
def magnitude(x):
    if x < 0:
        return -x
    else:
        return x
def init_delta_list():
    loss_delta_list = [4, 2, 3, 1]
def delta_list_pushback(value):
    d = loss_delta_list
    d[0] = d[1]
    d[1] = d[2]
    d[2] = d[3]
    d[3] = value

from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]
my_transforms = transforms.Compose([
    #transforms.ToPILImage(),  # needed if your image tensors are not PIL Images
    #transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
ldl = loss_delta_list


# I only want to train on members
print("Start Training")

trained = 0
while trained < amt_models and True:
    print(f"Shadow model {trained + 1}:")
    
    print("Build architecture.")
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 44)
    
    print("Set training configurations.")
    
    model.to(device)
    
    dataset = TaskDataset(transform=my_transforms)
    train_data = shadow_partitions[trained][0]
    if not train_data[0].sample_is_member():
        print("Training data should be member.")
        exit()
        
    _ids = []
    images = []
    labels = []
    for w in train_data:
        _ids.append(w._id)
        images.append(w.img)
        labels.append(w.label)
    dataset.ids = _ids
    dataset.imgs = images
    dataset.labels = labels
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training shadow model {trained + 1}.")
    for epoch in range(epochs):
        model.train()
        init_delta_list()
        running_loss = 0
        for _ids, images, labels in loader: # batch_size determines runtime here
            images, labels = images.to(device), labels.to(device)
        
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        avg_loss = running_loss / len(loader)
        delta_list_pushback(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        if avg_loss <= loss_abort:
            print("ok")
            break
        if magnitude(ldl[0] - ldl[1]) <= loss_delta_abort and magnitude(ldl[1] - ldl[2]) <= loss_delta_abort and magnitude(ldl[2] - ldl[3]) <= loss_delta_abort:
            print("early stop, ok")
            break
            
    print(f"Saving shadow model {trained + 1}.")
    torch.save(model.state_dict(), f"{k.shadow_models_path}{k.shadow_name_prefix}{trained + 1}.pt") # save model weights 
    
    trained += 1
    print()
    
print("Go back to the shadow!")

