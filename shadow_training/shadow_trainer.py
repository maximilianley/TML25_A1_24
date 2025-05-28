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

file_path = ""
label_is_member = 1
label_is_not_member = 0

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


### Add this as a transofrmation to pre-process the images .. to normalize them
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_pub: MembershipDataset = torch.load("../pub.pt")
_ids = []
images = []
labels = []
for s in data_pub:
    if s[3] == label_is_member:
        _ids.append(s[0])
        labels.append(s[2]) # NOT s[3] you git !!!
        images.append(s[1])

# apply image transforms
my_transforms = transforms.Compose([
    #transforms.ToPILImage(),  # needed if your image tensors are not PIL Images
    #transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# dataset = TensorDataset(images, labels)
dataset = TaskDataset(transform=my_transforms)
dataset.ids = _ids
dataset.imgs = images
dataset.labels = labels
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Train")
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for _ids, images, labels in loader:
        running_loss = 0
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
torch.save(model.state_dict(), "./shadow_models/test_dir.pt") # save model weights

print("Fertig")


