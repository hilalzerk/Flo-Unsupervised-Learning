# #############################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu - FLO
# #############################################

# #############################################
# İş Problemi
# #############################################

# FLO müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor. Buna yönelik
# olarak müşterilerin davranışları tanımlanacak ve bu
# davranışlardaki öbeklenmelere göre gruplar oluşturulacak.


# #############################################
# Veri Seti Hikayesi
# #############################################

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# --- VERİ SETİ DEĞİŞKEN AÇIKLAMALARI ---

# | Değişken Adı                       | Açıklama                                                              |
# |:-----------------------------------|:----------------------------------------------------------------------|
# | master_id                          | Eşsiz müşteri numarası                                                |
# | order_channel                      | Alışveriş yapılan platforma ait kanal (Android, ios, Desktop, Mobile) |
# | last_order_channel                 | En son alışverişin yapıldığı kanal                                    |
# | first_order_date                   | Müşterinin yaptığı ilk alışveriş tarihi                               |
# | last_order_date                    | Müşterinin yaptığı son alışveriş tarihi                               |
# | last_order_date_online             | Müşterinin online platformda yaptığı son alışveriş tarihi             |
# | last_order_date_offline            | Müşterinin offline platformda yaptığı son alışveriş tarihi            |
# | order_num_total_ever_online        | Müşterinin online platformda yaptığı toplam alışveriş sayısı          |
# | order_num_total_ever_offline       | Müşterinin offline'da yaptığı toplam alışveriş sayısı                 |
# | customer_value_total_ever_offline  | Müşterinin offline alışverişlerinde ödediği toplam ücret              |
# | customer_value_total_ever_online   | Müşterinin online alışverişlerinde ödediği toplam ücret               |
# | interested_in_categories_12        | Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi        |

# #############################################
# Proje Görevleri
# #############################################

# Görev 1: Veriyi Hazırlama
# Adım 1: flo_data_20K.csv verisini okutunuz.
# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
# #############################################

# Görev 2: K-Means ile Müşteri Segmentasyonu
# Adım 1: Değişkenleri standartlaştırınız.
# Adım 2: Optimum küme sayısını belirleyiniz.
# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
# #############################################

# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.

# ##################################################################################################

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Görev 1: Veriyi Hazırlama

# Adım 1: flo_data_20K.csv verisini okutunuz.

df = pd.read_csv("Datasets/flo_data_20k.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1],numeric_only=True).T)


check_df(df)

# Tarih içeren sütunları datetime formatına çevirme

date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Kontrol
df.dtypes

# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.

# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.


# Toplam alışveriş sayısını ve toplam harcamayı hesaplama
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Analiz tarihini belirleme (Son alışverişten 2 gün sonrası standarttır)
today_date = df["last_order_date"].max() + dt.timedelta(days=2)

# Recency, Tenure ve Frequency değerlerini oluşturma
# Recency: Müşterinin son alışverişinden bugüne geçen zaman
df["recency"] = (today_date - df["last_order_date"]).dt.days

# Tenure: Müşterinin ilk alışverişinden bugüne geçen zaman (Müşterinin yaşı)
df["tenure"] = (today_date - df["first_order_date"]).dt.days

# Segmentasyon için kullanılacak DataFrame'i hazırlama
# Genelde segmentasyon için: Recency, Frequency (Toplam Alışveriş) ve Monetary (Toplam Ücret) kullanılır.

model_df = df[["master_id", "recency", "tenure", "order_num_total", "customer_value_total"]]

model_df.head()

model_df.describe().T

# Aykırı değerlerin eşik değerlerini belirleyen fonksiyon
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05) # Alt eşik için %1
    quartile3 = dataframe[variable].quantile(0.95) # Üst eşik için %99
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değerleri eşik değerlerle baskılayan (değiştiren) fonksiyon
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

for col in ["order_num_total", "customer_value_total"]:
    replace_with_thresholds(model_df, col)

model_df.describe().T

# Görev 2: K-Means ile Müşteri Segmentasyonu
# Adım 1: Değişkenleri standartlaştırınız.

# Sadece modelleme için gerekli sütunları bir listeye alalım
segmentation_cols = ["recency", "tenure", "order_num_total", "customer_value_total"]
model_df = df[segmentation_cols]

# MinMaxScaler uyguluyoruz
sc = MinMaxScaler((0, 1))
# fit_transform sonrası veriyi tekrar DataFrame'e çeviriyoruz
model_df_scaled = pd.DataFrame(sc.fit_transform(model_df), columns=model_df.columns)

model_df_scaled

# Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(model_df_scaled)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df_scaled)
elbow.show()

elbow.elbow_value_

# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=42).fit(model_df_scaled)

clusters_kmeans = kmeans.labels_

df["kmeans_cluster"] = clusters_kmeans

df["kmeans_cluster"] = df["kmeans_cluster"]+ 1

df[["master_id", "kmeans_cluster"] + segmentation_cols].head()

# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.

k_means_analysis = df.groupby("kmeans_cluster").agg({
    "recency": ["mean", "count"],
    "tenure": ["mean"],
    "order_num_total": ["mean"],
    "customer_value_total": ["mean"]
}).sort_values(by=("customer_value_total", "mean"), ascending=False)

k_means_analysis

plt.figure(figsize=(8, 5))
sns.countplot(x="kmeans_cluster", data=df)
plt.title("Segmentlere Göre Müşteri Dağılımı")
plt.show()

# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

model_df_scaled

hc_complete = linkage(model_df_scaled, "ward")

# 2. Dendrogram grafiğini çizdirme
plt.figure(figsize=(15, 8))
plt.title("Hiyerarşik Kümeleme Dendrogramı")
plt.xlabel("Gözlem Birimleri (Son 10 Küme)")
plt.ylabel("Uzaklıklar")

# truncate_mode="lastp" ile karmaşıklığı önlemek için sadece son p kümeyi gösteriyoruz
dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)

# Kırılma noktasını belirlemek için yatay bir çizgi çekebiliriz
plt.axhline(y=12, color='r', linestyle='--')
plt.show()


# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.

from sklearn.cluster import AgglomerativeClustering

hi_cluster = AgglomerativeClustering(n_clusters=5, linkage="ward")

hi_clusters = hi_cluster.fit_predict(model_df_scaled)

df["hi_cluster_no"] = hi_clusters+ 1

df[["master_id", "hi_cluster_no"]].head()

df[["master_id", "kmeans_cluster", "hi_cluster_no"] + segmentation_cols].head()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.

hi_analysis = df.groupby("hi_cluster_no").agg({
    "recency": ["mean", "count"],
    "tenure": ["mean"],
    "order_num_total": ["mean"],
    "customer_value_total": ["mean"]
}).sort_values(by=("customer_value_total", "mean"), ascending=False)

hi_analysis



########################################


# İki şartı da sağlayan müşterileri getir
farkli_kümelenenler = df[(df["kmeans_cluster"] == 6) & (df["hi_cluster_no"] == 2)]


farkli_kümelenenler


musteri_detay = df[df["master_id"] == "b5625f4e-a151-11eb-a568-000d3a38a36f"]
musteri_detay


musteri_detay = df[df["master_id"] == "c2e15af2-9eed-11e9-9897-000d3a38a36f"]
musteri_detay






