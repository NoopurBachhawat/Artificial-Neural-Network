import numpy as np

def bipolar(x):
    return np.where(x == 0, -1, 1)

def predict(weights, X):
    s = np.dot(X, weights)
    return np.where(s >= 0, 1, -1)

def train_hebb_verbose(X, y, lr=0.1, epochs=5):
    w = np.zeros(X.shape[1])  
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}")
        for xi, yi in zip(X, y):
            w += lr * yi * xi  
            print(f"Input: {xi}, Target: {yi}, Updated Weights: {w}")
        preds = predict(w, X)
        preds01 = np.where(preds == -1, 0, 1)
        print(f"Predictions after epoch {epoch}: {preds01.tolist()}")
    return w

X = np.array([
    [1, 0, 0], 
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

choice = input("Enter gate (AND/OR): ").strip().upper()

if choice == "AND":
    targets = np.array([0, 0, 0, 1])
elif choice == "OR":
    targets = np.array([0, 1, 1, 1])
else:
    print("Invalid choice! Please enter AND or OR.")
    exit()

y = bipolar(targets)

final_w = train_hebb_verbose(X, y, lr=0.2, epochs=5)

print("\nFinal Weights:", final_w)
