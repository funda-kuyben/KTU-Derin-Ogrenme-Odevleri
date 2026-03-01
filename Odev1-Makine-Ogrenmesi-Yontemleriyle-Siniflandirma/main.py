import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 1) Veri okuma
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# 2) Gereksiz sütunları sil (varsa)
for col in ["id", "Unnamed: 32"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# 3) Hedef sütun dönüştürme (M=1, B=0)
df["diagnosis"] = df["diagnosis"].astype(str).str.strip().map({"M": 1, "B": 0})

# Özellikler ve hedef
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 4) Eğitim / Test bölme (%80 / %20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Ölçekleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6) Model (Logistic Regression)
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# 7) Tahmin
y_prediction = model.predict(X_test)

# 8) Metrikler
accuracy = accuracy_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
recall = recall_score(y_test, y_prediction)
f1 = f1_score(y_test, y_prediction)

cm = confusion_matrix(y_test, y_prediction)  # [[TN FP],[FN TP]]
TN, FP, FN, TP = cm.ravel()

# 9) Sonuçlar
print("\n=== ODEV 1: Meme Kanseri Siniflandirma ===")
print("Veri kaynagi: data.csv")
print("Pozitif sinif: Malignant (1)\n")

print("Metrikler (Test Seti):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\nConfusion Matrix [[TN FP],[FN TP]]:")
print(cm)

print("\nDetay:")
print(f"TP (Malignant dogru)      : {TP}")
print(f"TN (Benign dogru)         : {TN}")
print(f"FP (Benign iken Malignant): {FP}")
print(f"FN (Malignant iken Benign): {FN}")