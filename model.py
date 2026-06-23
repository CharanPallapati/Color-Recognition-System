import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
 

X = []
y = []

# Red
for r in range(150, 256, 10):
    for g in range(0, 100, 20):
        for b in range(0, 100, 20):
            X.append([r, g, b])
            y.append("red")

# Green
for g in range(150, 256, 10):
    for r in range(0, 100, 20):
        for b in range(0, 100, 20):
            X.append([r, g, b])
            y.append("green")

# Blue
for b in range(150, 256, 10):
    for r in range(0, 100, 20):
        for g in range(0, 100, 20):
            X.append([r, g, b])
            y.append("blue")

# Yellow
for r in range(180, 256, 10):
    for g in range(180, 256, 10):
        for b in range(0, 80, 20):
            X.append([r, g, b])
            y.append("yellow")

# Orange
for r in range(180, 256, 10):
    for g in range(80, 180, 10):
        for b in range(0, 80, 20):
            X.append([r, g, b])
            y.append("orange")

# Black
for r in range(0, 60, 10):
    for g in range(0, 60, 10):
        for b in range(0, 60, 10):
            X.append([r, g, b])
            y.append("black")

# White
for r in range(200, 256, 10):
    for g in range(200, 256, 10):
        for b in range(200, 256, 10):
            X.append([r, g, b])
            y.append("white")
# Pink
for r in range(200, 256, 10):
    for g in range(100, 200, 10):
        for b in range(150, 256, 10):
            X.append([r, g, b])
            y.append("pink")

 
for r in range(120, 220, 10):
    for g in range(0, 100, 20):
        for b in range(180, 256, 10):
            X.append([r, g, b])
            y.append("violet")


X = np.array(X)
y = np.array(y)

print(X.shape)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

 

# Save model
with open("color_knn.pkl", "wb") as f:
    pickle.dump(knn, f)