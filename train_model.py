from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Muat dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Kita akan latih model menggunakan SELURUH data agar lebih pintar
# (Praktik umum setelah menemukan parameter terbaik)
# Namun, untuk konsistensi dengan langkah tuning, kita latih di data train saja.
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Buat model final dengan k=3 (atau k terbaik dari langkah 1)
K_FINAL = 3
final_knn = KNeighborsClassifier(n_neighbors=K_FINAL)
final_knn.fit(X_train, y_train)

# Simpan model ke dalam sebuah file
nama_file_model = 'model_iris_knn.joblib'
joblib.dump(final_knn, nama_file_model)

print(f"Model telah berhasil dilatih dengan k={K_FINAL} dan disimpan sebagai '{nama_file_model}'")