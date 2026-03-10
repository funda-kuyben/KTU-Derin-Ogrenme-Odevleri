import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# XOR veri seti
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])


print("=== ODEV 2: XOR Probleminin Yapay Sinir Aglariyla Cozulmesi ===\n")

print("XOR Veri Seti:")
print("x1  x2  hedef")
for i in range(len(X)):
    print(f"{X[i][0]}   {X[i][1]}    {y[i]}")


# 1) Tek katmanli perceptron
single_model = Perceptron(max_iter=1000, random_state=42)
single_model.fit(X, y)
single_prediction = single_model.predict(X)
single_accuracy = accuracy_score(y, single_prediction)

print("\n--- Tek Katmanli Perceptron Sonuclari ---")
print("x1  x2  hedef  tahmin")
for i in range(len(X)):
    print(f"{X[i][0]}   {X[i][1]}    {y[i]}      {single_prediction[i]}")
print(f"Accuracy: {single_accuracy:.4f}")


# 2) Cok katmanli yapay sinir agi
multi_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)
multi_model.fit(X, y)
multi_prediction = multi_model.predict(X)
multi_accuracy = accuracy_score(y, multi_prediction)

print("\n--- Cok Katmanli Yapay Sinir Agi Sonuclari ---")
print("x1  x2  hedef  tahmin")
for i in range(len(X)):
    print(f"{X[i][0]}   {X[i][1]}    {y[i]}      {multi_prediction[i]}")
print(f"Accuracy: {multi_accuracy:.4f}")


# 3) Genel yorum
print("\n--- Deney Sonucu ---")
if single_accuracy < 1.0:
    print("Tek katmanli perceptron XOR problemini tam olarak cozememistir.")
else:
    print("Tek katmanli perceptron XOR problemini cozmustur.")

if multi_accuracy == 1.0:
    print("Cok katmanli yapay sinir agi XOR problemini basariyla ogrenmistir.")
else:
    print("Cok katmanli yapay sinir agi XOR problemini tam olarak ogrenememistir.")