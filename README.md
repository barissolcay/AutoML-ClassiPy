# ML Classification System

Makine öğrenmesi algoritmalarını kullanarak sınıflandırma yapan bir Python uygulaması.

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- **Python 3.10+** ([İndir](https://www.python.org/downloads/))

### 1. Kütüphaneleri Yükleyin

Komut istemcisini (CMD veya PowerShell) açın ve proje klasörüne gidin:

```bash
cd "proje_klasoru_yolu"
pip install -r requirements.txt
```

Veya manuel yükleme:

```bash
pip install pandas scikit-learn joblib
```

### 2. Uygulamayı Çalıştırın

```bash
python app/gui_main.py
```

---

## 📖 Kullanım

1. **Dataset Selection**: "Browse" butonuna tıklayın ve bir CSV dosyası seçin
2. **Load Dataset**: Veri setini yükleyin
3. **Discover Best Algorithm**: Tüm algoritmaları test edin. En iyi algoritma otomatik seçilir
4. **Prediction**: Yeni veriler için tahmin yapın

---

## 📂 Proje Yapısı

```
decision_system/
├── app/
│   └── gui_main.py          # Ana GUI uygulaması
├── core/
│   ├── dataset_loader.py    # CSV yükleme ve feature tespiti
│   ├── model_definition.py  # 11 farklı ML algoritması tanımları
│   ├── model_trainer.py     # Model eğitimi ve 10-fold cross-validation
│   ├── preprocessing.py     # Veri ön işleme (scaling, encoding)
│   ├── best_model_manager.py# En iyi model yönetimi
│   └── model_result.py      # Sonuç veri yapısı
├── heart.csv                 # Örnek veri seti
├── requirements.txt          # Bağımlılıklar
└── README.md                 # Bu dosya
```

---

## 🧠 Kullanılan Algoritmalar

| # | Algoritma | Açıklama |
|---|-----------|----------|
| 1 | Naive Bayes | Discretized features ile |
| 2 | Logistic Regression | Scaled features ile |
| 3 | KNN (k=1) | En yakın 1 komşu |
| 4 | KNN (k=3) | En yakın 3 komşu |
| 5 | KNN (k=5) | En yakın 5 komşu |
| 6 | Decision Tree | J48 benzeri |
| 7 | Random Forest | 100 ağaç ile |
| 8 | Extra Trees | 100 ağaç ile |
| 9 | MLP (Neural Network) | 100 nöronlu 1 gizli katman |
| 10 | SVM (Linear) | Doğrusal kernel |
| 11 | SVM (RBF) | Radial basis function kernel |

---

## ⚙️ Değerlendirme Yöntemi

- **10-Fold Stratified Cross-Validation**
- Her algoritma için doğru sınıflandırma sayısı ve accuracy hesaplanır
- En yüksek accuracy'ye sahip model otomatik seçilir ve yeni tahminler için kullanılır
