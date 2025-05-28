import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
from random import randint
import __main__

#### LOADING THE MODEL

from torchvision.models import resnet18

# torch.serialization.add_safe_globals([__main__.MembershipDataset])

### Add this as a transofrmation to pre-process the images .. to normalize them
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA.pt", map_location="cpu") # , weights_only = False)

model.load_state_dict(ckpt)



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

# torch.serialization.add_safe_globals([MembershipDataset])
# torch.serialization.add_safe_globals({"__main__.MembershipDataset": MembershipDataset}) # extra line to give explicit permissions for the "MembershipDataset" (allowlisting)
# __module__ = "__main__"
torch.serialization.add_safe_globals([__main__.MembershipDataset]) # you can do it only AFTER you declared the class ... OHHH MYYY GOOOODDDD
# Force-register using the internal expected name
"""print(f"The Membership module name: {MembershipDataset.__module__}.{MembershipDataset.__qualname__}")
torch.serialization.add_safe_globals({
    f"{MembershipDataset.__module__}.{MembershipDataset.__qualname__}": MembershipDataset
})"""


print()
data: MembershipDataset = torch.load("./priv_out.pt") # , weights_only = False)
print("Private data sample inspection")
probe_index = 12 # 71 # 45 high confidence , 33
print(type(data[probe_index]))
print(str(data[probe_index]))



print()
data_pub: MembershipDataset = torch.load("./pub.pt") # , weights_only = False)
print("public data sample inspection")
probe_index_pub = 12 # 45 high confidence , 33
pub_data_sample = data_pub[probe_index_pub]
print(type(pub_data_sample))
print(str(pub_data_sample))
member_idxs = []
the_little_squishy_duck = 6
counter = 0
for irish_pub in data_pub:
    if irish_pub[3] == 1:
        member_idxs.append(counter)
        if len(member_idxs) >= the_little_squishy_duck:
            break
    counter += 1
print(member_idxs)



#### TEST MODEL
print("test model")
model.eval()
pub_members_amt = 0
conf_priv_array = []
with torch.no_grad():
    print(f"id = {data_pub[probe_index_pub][0]}")
    ok_sample = data_pub[probe_index_pub][1].unsqueeze(0)
    output = model(ok_sample)
    print("Oh, you touch my Tralala")
    print(str(output))
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Confidence: {confidence.item()}")
    
    counter = 0
    confident_treshld = 0.96
    if False:
        for irish_pub in data_pub:
            pub_ok_sample = irish_pub[1].unsqueeze(0)
            ite_out = model(pub_ok_sample)
            probs = torch.nn.functional.softmax(ite_out, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            if confidence.item() > confident_treshld:
                counter += 1
        pub_members_amt = counter
        
    if True:
        r_amount = 1500
        scalar = 0.0001
        for sample in data:
            pub_ok_sample = sample[1].unsqueeze(0)
            ite_out = model(pub_ok_sample)
            probs = torch.nn.functional.softmax(ite_out, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            if confidence.item() > confident_treshld:
                faked_good_confidence = 0.99 - scalar * randint(0, r_amount)
                conf_priv_array.append(np.log(faked_good_confidence/(1.0 - faked_good_confidence)))
                # counter += 1
            else:
                faked_bad_confidence = 0.01 + scalar * randint(0, r_amount)
                conf_priv_array.append(np.log(faked_bad_confidence/(1.0 - faked_bad_confidence)))
        pub_members_amt = counter
    
    counter = 0
    if False:
        print("Running batch through model")
        outputs = model(data_pub[10:20]) # doesn't work
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probs, dim=1)
        for i in range(len(data_pub)):
            if confidences[i].item() > confident_treshld:
                counter += 1
        pub_members_amt = counter

    
#### EXAMPLE SUBMISSION

random_samples = np.random.randn(len(data.ids))
print(str(random_samples))
print("Random samples")
print(type(random_samples))
print(type(random_samples[8]))

my_confs = np.array(conf_priv_array)
print(str(my_confs))
df = pd.DataFrame(
    {
        "ids": data.ids,
        "score": my_confs, # random_samples,
    }
)
df.to_csv("test.csv", index=None)


print()
print("How big are deezz datasets?")
print(f"The private set: {len(data)}")
print(f"The irish set: {len(data_pub)}")
print()
print("How many of public are members?")
mem_count = 0
for yo_mama in data_pub:
    if yo_mama[3] == 1:
        mem_count += 1
print(f"{mem_count}  --  percentage: {mem_count/len(data_pub)}")
print(f"And filtered by threshold: {confident_treshld}  --  percentage: {pub_members_amt/len(data_pub)}")
print()



def send_request():
    try:
        response = requests.post("http://34.122.51.94:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "61002325"})
        print(response.json())
    except:
        print("sth went WRONG MAAAAAANNN.")
    
#send_request()
