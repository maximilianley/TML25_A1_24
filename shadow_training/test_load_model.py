import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
from random import randint
from torchvision.models import resnet18
import __main__

print("Try loading model.")

# recreate the same architecture (very important)
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

#ckpt = torch.load("./test_model_weights.pt", map_location="cpu")
ckpt = torch.load("./shadow_models/shadow_7.pt", map_location="cpu")

model.load_state_dict(ckpt)
print("Model loaded successfully.")


print("Load public dataset")

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

print("Evaluate model with some sample.")

model.eval()
probe_index_pub = 12
with torch.no_grad():
    print(f"id = {data_pub[probe_index_pub][0]}")
    print(f"correct label = {data_pub[probe_index_pub][2]}")
    print("sample:")
    print(str(data_pub[probe_index_pub][1]))
    ok_sample = data_pub[probe_index_pub][1].unsqueeze(0)
    output = model(ok_sample)
    print(str(output))
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Confidence: {confidence.item()}")
