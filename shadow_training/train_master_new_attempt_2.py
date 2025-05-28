import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple
import __main__

        
        
        
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





class MetaClassifier(nn.Module):
    def __init__(self, in_dims = 44):
        super(MetaClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dims, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2) # Binary classification: [yes, no]
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # No softmax here, use CrossEntropyLoss
        
        
        
        

shadow_pub: TaskDataset = torch.load("./evaluated_shadow_pub_shuffled.pt")
confidences = torch.stack(shadow_pub.imgs).squeeze(1)
labels = torch.tensor(shadow_pub.labels)
#dataset = TensorDataset(confidences, labels)





print(f"torch.bincount: {torch.bincount(labels)}")
entropy = -torch.sum(confidences * torch.log(confidences + 1e-8), dim=1, keepdim=True)
max_conf = torch.max(confidences, dim=1, keepdim=True).values
features = torch.cat([confidences, entropy, max_conf], dim=1)
mean = features.mean(dim=0, keepdim=True)
std = features.std(dim=0, keepdim=True)
std[std == 0] = 1.0 # Avoid division by zero
features = (features - mean) / std
features_scaled = features
#print("entropy: " + str(entropy))
#print("max_conf: " + str(max_conf))
#print("features: " + str(features))
print(str(features.shape[1]))
input_dimensions = features.shape[1]
#exit()
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"loader length: {len(loader)}")
meta_model = MetaClassifier(in_dims = input_dimensions)
optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()



meta_model.train() # muss nicht vorhanden sein
for epoch in range(40):
    total_loss = 0
    for x_batch, y_batch in loader:
        logits = meta_model(x_batch)
        loss = criterion(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Avg. Loss: {(total_loss/len(loader)):.4f}")

# starts to overfit -> loss starts to jump around in a local minimum .. in need of a bigger network ?-> more room for expression
# too big of a step (gradient) jumps around right from the start

    
    
    
    
# eval accuracy
meta_model.eval()
with torch.no_grad():
    logits = meta_model(features_scaled)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean()
    print(f"Meta-classifier accuracy: {acc:.4f}")

