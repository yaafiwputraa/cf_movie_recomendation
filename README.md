

# Collaborative Filtering Movie Recommendation

Sistem rekomendasi film menggunakan collaborative filtering dengan PyTorch.

## Deskripsi Project

Project ini mengimplementasikan sistem rekomendasi film menggunakan teknik collaborative filtering dengan matrix factorization. Sistem dapat memprediksi rating film untuk user berdasarkan pola rating yang sudah ada.

## Fitur

- Matrix factorization untuk collaborative filtering
- Implementasi menggunakan PyTorch
- Normalisasi rating untuk performa yang lebih baik
- Cost function dengan regularisasi L2
- Optimasi menggunakan Adam optimizer

## Struktur Kode

### 1. Pembuatan Dataset
```python
def create_dataset(ratings):
    matrix = ratings.pivot(index="movieId", columns="userId", values="rating")
    matrix = matrix.fillna(0)
    return matrix
```

### 2. Cost Function
```python
def cost_func(X, W, b, Y, R, lambda_):
    y_hat = torch.matmul(X,W.t()) + b 
    j = (y_hat - Y) * R
    J = 0.5* torch.sum(j**2) + lambda_/2 * (torch.sum(W**2) + torch.sum(X**2)) 
    return J
```

### 3. Normalisasi Rating
```python
def normalizeRatings(Y, R):
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)
```

## Cara Penggunaan

1. Load data ratings dan movies
2. Buat matrix rating menggunakan `create_dataset()`
3. Buat matrix indikator R (1 jika ada rating, 0 jika tidak)
4. Tambahkan rating untuk user baru
5. Siapkan data untuk training dengan `prepare_train()`
6. Train model menggunakan Adam optimizer
7. Dapatkan prediksi dan rekomendasi

### Contoh Penggunaan

```python
# Buat dataset
Y_df = create_dataset(ratings)
R_df = (Y_df != 0).astype(int)

# Tambahkan rating user baru
my_ratings = np.zeros(Y_df.shape[0])
my_ratings[929] = 4   # Contoh: rating 4 untuk film dengan index 929

# Training
X, W, b, R, Ynorm, Ymean = prepare_train(Y_df, R_df, my_ratings, num_features=150)
opt = optim.Adam([X,W,b], lr=1e-1)
train(200, lambda_=1)

# Prediksi
pred = torch.matmul(X, W.t()) + b
predm = pred + Ymean
my_pred = predm[:, -1]  # Prediksi untuk user baru
```

## Parameter Model

- `num_features`: Jumlah fitur laten (default: 150)
- `learning_rate`: Learning rate untuk Adam optimizer (default: 0.1)
- `lambda_`: Parameter regularisasi (default: 1)
- `iterations`: Jumlah iterasi training (default: 200)

## Hasil

Model akan menghasilkan:
- Prediksi rating untuk film yang belum ditonton
- Rekomendasi film berdasarkan prediksi rating tertinggi
- Perbandingan rating asli vs prediksi untuk validasi