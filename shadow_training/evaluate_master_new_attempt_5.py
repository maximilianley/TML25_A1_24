import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, in_dims = 44):
        super(MetaClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dims, 64)
        self.fc2 = nn.Linear(64, 96)
        self.fc3 = nn.Linear(96, 128)
        self.fc4 = nn.Linear(128, 96)
        self.fc5 = nn.Linear(96, 64)
        #self.bn3 = nn.BatchNorm1d(64)
        #self.dropout3 = nn.Dropout(0.3)
        self.fc6 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2) # Binary classification: [yes, no]
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.out(x)  # No softmax here, use CrossEntropyLoss
        

print("Load private dataset")
data_priv: MembershipDataset = torch.load("../priv_out.pt")


# Harlem Shake

print("Load master model.")
model_master = MetaClassifier(in_dims = 46)
ckpt = torch.load("./master_linear_44_64_96_128_96_64_32_2.pt", map_location="cpu")
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

images = torch.stack(data_priv.imgs).squeeze(1)
labels = torch.tensor(data_priv.labels)
confs = None
with torch.no_grad():
    logits = model_target(images)
    probs = torch.nn.functional.softmax(logits, dim=1) # print(type(probs)) print(str(probs.shape)) print(str(probs[13]))
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
    max_conf = torch.max(probs, dim=1, keepdim=True).values
    features = torch.cat([probs, entropy, max_conf], dim=1)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std == 0] = 1.0 # Avoid division by zero
    features_scaled = (features - mean) / std
    logits = model_master(features_scaled)
    confs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean()
    print(str(preds))
    print(str(confs))
    print(str(data_priv.ids[0:10]))
    print(str(labels))
    print(f"Meta-classifier accuracy: {acc:.4f}")
    




# prepare and send
extracted_confs = [] # only need confidences at index 1
#print(str(confs[9][1]))
for f in confs:
    extracted_confs.append(f[1])
final_confs = np.array(extracted_confs)
print(len(final_confs))
print(type(final_confs))
print(str(final_confs[9]))
print(type(final_confs[9]))


df = pd.DataFrame(
    {
        "ids": data_priv.ids,
        "score": final_confs, # random_samples,
    }
)
df.to_csv("test.csv", index=None)

def send_request():
    try:
        print("Sending to server.")
        response = requests.post("http://34.122.51.94:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "61002325"})
        print(response.json())
    except:
        print("No internet, or some other network problem.")
    
send_request()





# eval accuracy of shadow_pub
shadow_pub: TaskDataset = torch.load("./evaluated_shadow_pub_shuffled.pt")
confidences = torch.stack(shadow_pub.imgs).squeeze(1)
labels = torch.tensor(shadow_pub.labels)
entropy = -torch.sum(confidences * torch.log(confidences + 1e-8), dim=1, keepdim=True)
max_conf = torch.max(confidences, dim=1, keepdim=True).values
features = torch.cat([confidences, entropy, max_conf], dim=1)
mean = features.mean(dim=0, keepdim=True)
std = features.std(dim=0, keepdim=True)
std[std == 0] = 1.0 # Avoid division by zero
features_scaled = (features - mean) / std

with torch.no_grad():
    logits = model_master(features_scaled)
    preds = torch.argmax(logits, dim=1)
    #preds = preds + (-1)
    acc = (preds == labels).float().mean()
    print(f"Meta-classifier accuracy: {acc:.4f}")


