import pickle

with open("priv_out.pt", "rb") as f:
    raw = pickle.load(f) # , fix_imports=False)
print(type(raw))
print(dir(raw))

