import pickle

f = open("dataset.pkl", "rb")
data = pickle.load(f, encoding="bytes")
print(str(data[1]))
