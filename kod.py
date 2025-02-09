# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:30:01 2024

@author: Ataka
"""

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np

# Kaggle API anahtarını kaydetme
kaggle_dir = os.path.expanduser("C:/Users/Ataka/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

api_key = {
    "username": "atakanclskn",
    "key": "apikey"
}

with open(os.path.join(kaggle_dir, "kaggle.json directory path yazin lutfen"), "w") as f:
    import json
    json.dump(api_key, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# Kaggle API ile veri setini indirme
api = KaggleApi()
api.authenticate()

dataset = "farukalam/yelp-restaurant-reviews"
download_path = "yelp_reviews_data"
api.dataset_download_files(dataset, path=r'C:\Users\Ataka\OneDrive\Masaüstü\ODEV', unzip=True)

print(f"Dataset indirildi ve {download_path} klasörüne çıkarıldı.")

# Veriyi yükleme
df = pd.read_csv(r'C:\Users\Ataka\OneDrive\Masaüstü\ODEV\Yelp Restaurant Reviews.csv')
print("Veri Seti Başlıkları:", df.columns)
print("Veri Seti İlk 5 Satır:\n", df.head())

# Rating sütunundaki etiketleri 0'dan başlayacak şekilde yeniden sıralama xgboost icin bunu yaptim
df['Rating'] = df['Rating'] - 1  

# Yorumları tokenizasyon yapma
df['tokens'] = df['Review Text'].apply(lambda x: word_tokenize(str(x).lower()))

# Word2Vec  eğitme
model = Word2Vec(df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Yorumları vektörleştirme
def vectorize_review(tokens):
    return [model.wv[word] for word in tokens if word in model.wv]

# Tüm yorumları vektörleştirme
X = df['tokens'].apply(lambda x: vectorize_review(x))

# Vektörlerin uzunluğunu eşitleme (padding veya average olabilir, burada ortalamayı kullanalım)
def average_vector(vectors):
    if len(vectors) == 0:
        return [0]*100  # Eğer boşsa sıfır vektörü döndürelim
    return sum(vectors) / len(vectors)

X = X.apply(lambda x: average_vector(x))

# Y alalım
y = df['Rating']

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(list(X), y, test_size=0.2, random_state=42)

# Logistic Regression 
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_accuracy)

# Random Forest 
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)

# XGBoost 
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # XGBoost ile label_encoder kullanımını kapattık
xgb.fit(X_train, y_train)  # Bu satırda XGBoost çalışacak
y_pred_xgb = xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", xgb_accuracy)

# Grafik çizme: Model doğruluklarının karşılaştırılması
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracies = [lr_accuracy, rf_accuracy, xgb_accuracy]

# Bar grafiği çizme
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Modeller')
plt.ylabel('Doğruluk Oranı')
plt.title('Modellerin Doğruluk Oranları')
plt.ylim(0, 1)  # Doğruluk oranları 0 ile 1 arasında olmalı
plt.show()

# Yeni yorumların tahmin edilmesi
new_reviews = [
    "Basically the best thing since sliced bread. Definitely worth the trip outside of Champaign",
    "Ice cream and milk shake were good there are a lot of varieties. Staff so not so friendly",
    #databeseden dolayi 3 yorum eklemeden saglikli bir sonuc alinmiyor. Buna gore bir cumle daha ekledim ve kodumu ona gore duzenledim.
    "Nice food and cozy atmosphere, but a bit expensive for the portion size"
]

# Yorum tokenizasyon ve vektörlere dönüştürme kismi
new_tokens = [word_tokenize(review.lower()) for review in new_reviews]
new_vectors = [vectorize_review(tokens) for tokens in new_tokens]

# Yeni yorumları tahmin etme
new_vectors_avg = [average_vector(vec) for vec in new_vectors]  # Ortalamalarını alalım

new_predictions_lr = lr.predict(new_vectors_avg)
new_predictions_rf = rf.predict(new_vectors_avg)
new_predictions_xgb = xgb.predict(new_vectors_avg)

# Yeni yorumların tahminleri
print("Logistic Regression Tahminleri:", new_predictions_lr)
print("Random Forest Tahminleri:", new_predictions_rf)
print("XGBoost Tahminleri:", new_predictions_xgb)

# Grafik çizme
labels = ['Review 1', 'Review 2', 'Review 3']
lr_predictions = new_predictions_lr
rf_predictions = new_predictions_rf
xgb_predictions = new_predictions_xgb

# Bar grafiği 
x = np.arange(len(labels))  # x eksenindeki pozisyonlar
width = 0.25  # Her bir çubuğun genişliği

fig, ax = plt.subplots(figsize=(10, 6))

# Barlar
ax.bar(x - width, lr_predictions, width, label='Logistic Regression', color='blue')
ax.bar(x, rf_predictions, width, label='Random Forest', color='green')
ax.bar(x + width, xgb_predictions, width, label='XGBoost', color='red')

# baslik
ax.set_xlabel('Yorumlar')
ax.set_ylabel('Tahmin Edilen Rating')
ax.set_title('Yeni Yorumların Tahmin Edilen Puanları (Model Karşılaştırması)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# diger Grafik
plt.tight_layout()
plt.show()
