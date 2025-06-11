# ---------------------------------------------------------------------------
# STUDI KASUS: KLASIFIKASI SPESIES BUNGA IRIS DENGAN KNN
# ---------------------------------------------------------------------------
# Kode ini tidak memerlukan file CSV eksternal.
# ---------------------------------------------------------------------------

# 1. MENGIMPOR LIBRARY
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Tahap 1: Library berhasil diimpor.")

# 2. MEMUAT DATASET IRIS
# Dataset Iris sudah tersedia di dalam library scikit-learn
iris = load_iris()
# Konversi ke DataFrame pandas agar lebih mudah dianalisis
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print("\nTahap 2: Dataset Iris berhasil dimuat.")
print("Fitur yang digunakan:", iris.feature_names)
print("Target klasifikasi:", iris.target_names)
print("\n5 baris pertama dari dataset Iris:")
print(df.head())


# (Opsional) Visualisasi data untuk melihat sebaran kelas
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle("Visualisasi Sebaran Data Iris", y=1.02)
plt.show()


# 3. PERSIAPAN DATA UNTUK MODEL
# Memisahkan fitur (X) dan target (y)
X = iris.data
y = iris.target

# Membagi dataset menjadi data training (80%) dan data testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTahap 3: Data telah dibagi menjadi data training dan testing.")
print(f"   - Jumlah data training: {len(X_train)}")
print(f"   - Jumlah data testing: {len(X_test)}")


# 4. PEMBUATAN DAN PELATIHAN MODEL KNN
# Membuat instance model KNN dengan k=3 (nilai k yang umum untuk dataset ini)
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model menggunakan data training
knn.fit(X_train, y_train)
print("\nTahap 4: Model KNN telah berhasil dilatih dengan k=3.")


# 5. MELAKUKAN PREDIKSI PADA DATA TESTING
y_pred = knn.predict(X_test)
print("Tahap 5: Prediksi pada data testing telah selesai.")


# 6. EVALUASI KINERJA MODEL
print("\n----------------- HASIL EVALUASI MODEL IRIS -----------------")

# Menghitung dan menampilkan akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy * 100:.2f}%\n")

# Menampilkan Laporan Klasifikasi (Precision, Recall, F1-Score)
# Kita gunakan target_names untuk menampilkan nama kelas (spesies)
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Menampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
# Membuat visualisasi confusion matrix agar lebih mudah dibaca
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix untuk Data Iris')
plt.show()

print("\n----------------------------------------------------------")
print("Eksekusi selesai.")