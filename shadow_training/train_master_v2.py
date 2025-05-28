import torch
from torch.utils.data import Dataset
from torch import optim
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import requests
import pandas as pd
from random import randint
from torchvision import transforms
import __main__
import model_constants as k


####### DATASETS ########

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
torch.serialization.add_safe_globals([__main__.TaskDataset])


###### CREATE THE MASTER (use as template for eval as well) #######
class MetaClassifier(nn.Module):
    def __init__(self):
        super(MetaClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(44, 64),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            #nn.BatchNorm1d(32),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 2)
            #nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
        
        
model = MetaClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for name, param in model.named_parameters():
    print(name)
print(model)

"""
nn.Linear(44, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)#,
"""

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')#, a=0.01)  # good for ReLU or even linear
        #nn.init.zeros_(m.bias)
        if m.bias is not None:
            print("ZZZZZZZZZZZ")
            #nn.init.zeros_(m.bias)
            nn.init.constant_(m.bias, 0.1*randint(1, 9))

model.apply(init_weights)

#### LOAD DATASETS
print("Load Dataset that the shadow models evaluated and which all comes from the pub.pt dataset.")
# shadow_pub: TaskDataset = torch.load("./evaluated_shadow_pub.pt")
shadow_pub: TaskDataset = torch.load("./evaluated_shadow_pub_shuffled.pt")

'''
print(str(shadow_pub[1][0]))
print(str(shadow_pub[2][0]))
print(str(shadow_pub[3][0]))
print(str(shadow_pub[4][0]))
print(str(shadow_pub[5][0]))
print(str(shadow_pub[6][0]))
print(str(shadow_pub[7][0]))
print(str(shadow_pub[8][0]))
print(str(shadow_pub[9][0]))
print(str(shadow_pub[10][0]))
print(str(shadow_pub[11][0]))
'''

'''
print(str(shadow_pub[3][1]))
print(type(shadow_pub[3][1][0][5]))
print(str(shadow_pub[3][1][0][5].values))
norm_conf = nn.functional.softmax(shadow_pub[3][1], dim=1)
print(str(norm_conf))
print(type(norm_conf[0][5]))
'''

#dataset = TaskDataset()
#dataset.ids = _ids
#dataset.imgs = images
#dataset.labels = labels
dataset = shadow_pub
print("Raw img/conf sample")
print(str(shadow_pub.imgs[55]))
tensored_labels = []
for x in shadow_pub.labels:
    tensored_labels.append(torch.stack([torch.stack([torch.tensor(x), torch.tensor(1 - x)]).float()]))
dataset = TensorDataset(torch.stack(shadow_pub.imgs), torch.stack(tensored_labels))
dataset = TensorDataset(torch.stack(shadow_pub.imgs).squeeze(1), torch.tensor(shadow_pub.labels).long())
print("Sample input shape:", dataset[0][0].shape)
print("Sample label:", dataset[0][1])
print("Label shape:", dataset[0][1].shape)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

#print(f"Sample output: {model(torch.randn(1, 44))}")  # Should be [1, 2]
print(f"Sample label: {torch.tensor(shadow_pub.labels[0])}")  # Should be 0 or 1

# loss function
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Train")
num_epochs = 900
ggp = 0
for epoch in range(num_epochs):
    model.train()
    #for _ids, images, labels in loader:
    running_loss = 0
    for images, labels in loader:
        ggp += 1
        optimizer.zero_grad()
        #running_loss = 0
        images = (images - images.mean(dim=0)) / images.std(dim=0)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #before = model.model[0].weight.clone().detach()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        #after = model.model[0].weight.clone().detach()
        if ggp % 200 == 0 and False:
            print("Weight changed:", not torch.equal(before, after))
        
        if ggp == 27300 and False: #6500: # 1305
            print(str(images.shape))
            print(str(images.mean().item()))
            print(str(images.std().item()))
            print(str(labels.shape))
            print(str(images[18]))
            print(str(outputs))
            print(str(labels))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            print(str(probs))
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: mean grad = {param.grad.mean().item()}, std = {param.grad.std().item()}")
                if 'bias' in name:
                    print(f"{name}: {param.data}")

            exit()

        running_loss += loss.item()
        
    avg_loss = running_loss / len(loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print(f"ggp = {ggp}")
        
torch.save(model.state_dict(), "./master_linear_44_64_32_2.pt") # save model weights

print(f"DKHDKHDJK ggp {ggp}")
print("Fertig")

