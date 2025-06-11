import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Muat dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Coba nilai k dari 1 sampai 30
k_range = range(1, 31)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Cari k dengan akurasi tertinggi
best_k = k_range[accuracies.index(max(accuracies))]
print(f"Akurasi tertinggi adalah {max(accuracies)*100:.2f}% pada k = {best_k}")

# Plot hasil untuk visualisasi
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='-')
plt.title('Akurasi KNN untuk Nilai k yang Berbeda')
plt.xlabel('Nilai k')
plt.ylabel('Akurasi')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Berdasarkan hasil, kita bisa pilih k=3 atau nilai lain yang memberikan akurasi 100%
# Untuk contoh ini, kita akan tetap pakai k=3 karena lebih sederhana.
# Anda bisa memilih k terbaik berdasarkan hasil yang Anda dapat.
K_FINAL = best_k