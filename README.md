# FLO Müşteri Segmentasyonu (Unsupervised Learning)
Bu proje, FLO'nun müşteri verilerini kullanarak Gözetimsiz Öğrenme (Unsupervised Learning) teknikleriyle müşteri segmentasyonu yapmayı amaçlar. Müşterilerin alışveriş davranışları (Recency, Tenure, Frequency, Monetary) analiz edilerek benzer davranış gösteren gruplar oluşturulmuş ve bu gruplar üzerinden pazarlama stratejileri geliştirilmesi hedeflenmiştir.

## 📊 Veri Seti Hikayesi
Veri seti, FLO’dan 2020 - 2021 yıllarında OmniChannel (hem online hem offline) alışveriş yapan müşterilerin geçmiş davranışlarından oluşmaktadır.

Değişken,Açıklama
master_id,Eşsiz müşteri numarası
order_channel,"Alışveriş yapılan platform kanalı (Android, iOS, Desktop, Mobile)"
last_order_date,En son alışveriş yapılan tarih
order_num_total,Toplam alışveriş sayısı (Online + Offline)
customer_value_total,Müşterinin toplam harcaması (Online + Offline)
interested_in_categories_12,Son 12 ayda alışveriş yapılan kategoriler

🛠️ Uygulanan Adımlar
1. Veri Hazırlama (Data Preparation)

Tarih değişkenleri datetime formatına çevrildi.

Recency (yenilik), Tenure (müşteri yaşı), Frequency (toplam işlem) ve Monetary (toplam değer) metrikleri oluşturuldu.

Aykırı değerler (outliers), belirlenen eşik değerlerle (0.05 ve 0.95 quantiles) baskılandı.

2. K-Means Segmentasyonu

Veriler MinMaxScaler ile standartlaştırıldı.

Elbow Yöntemi ve KElbowVisualizer kullanılarak optimum küme sayısı belirlendi.

Belirlenen küme sayısına göre model fit edildi ve her müşteriye bir segment atandı.

Segmentlerin istatistiksel özetleri (ortalama harcama, alışveriş sıklığı vb.) çıkarıldı.

3. Hiyerarşik Kümeleme (Hierarchical Clustering)

Standardize edilmiş veri seti üzerinden ward yöntemiyle mesafe matrisi oluşturuldu.

Dendrogram grafiği ile verinin hiyerarşik yapısı görselleştirildi.

AgglomerativeClustering kullanılarak müşteriler alternatif bir yöntemle segmente edildi.

🚀 Öne Çıkan Bulgular ve Analiz
Proje sonunda her iki modelin (K-Means ve Hierarchical) sonuçları karşılaştırılmıştır.

Müşteriler; harcama potansiyelleri, alışveriş sıklıkları ve markaya olan bağlılık sürelerine (Tenure) göre gruplandırılmıştır.

Örneğin; hem harcaması yüksek hem de en son alışveriş tarihi çok yakın olan "Şampiyon" segmenti ile uzun süredir alışveriş yapmayan "Riskli" segmentler ayırt edilmiştir.

## 💻 Gereksinimler
numpy
pandas
matplotlib
seaborn
scikit-learn
yellowbrick
scipy
