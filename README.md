# KTU Derin Ogrenme Odevleri

Bu repository Derin Ogrenme dersi kapsaminda yapilan odevleri icermektedir.

## Odev 1 - Makine Ogrenmesi Yontemleriyle Siniflandirma

Bu odevde meme kanseri veri seti uzerinde ikili siniflandirma uygulanmistir.

### Kullanilan Yontem
- Logistic Regression
- StandardScaler ile ozellik olceklendirme

### Degerlendirme Metrikleri
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Confusion Matrix

Test sonuclari:

Accuracy  : 0.9649  
Precision : 0.9750  
Recall    : 0.9286  
F1-Score  : 0.9512  

Confusion Matrix:

|        | Tahmin: Benign | Tahmin: Malignant |
|--------|----------------|-------------------|
| Gercek: Benign    | 71 | 1 |
| Gercek: Malignant | 3  | 39 |
