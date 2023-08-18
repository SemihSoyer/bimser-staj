# Gerekli kütüphaneleri içe aktar
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Streamlit Arayüzü
st.title('Streamlit ile K-Means Kümeleme')
st.write('Özellikleri seçin ve k-means kümelerini görselleştirin.')

# Veri Girişi
st.sidebar.header('Veri Girişi')
st.sidebar.write('Veri noktalarını girin:')
val1 = st.sidebar.number_input('val1', value=0)
val2 = st.sidebar.number_input('val2', value=0)
new_data = np.array([[val1, val2]])

# Örnek Veri Oluşturma
veri = np.random.rand(100, 2) * 10

# Küme Sayısı Seçimi
num_clusters = st.sidebar.slider('Küme sayısını seçin', min_value=2, max_value=10, value=3)

# K-Means Model Oluşturma ve Eğitme
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(veri)
etiketler = kmeans.predict(veri)
merkezler = kmeans.cluster_centers_

# Veri Çerçevesi Oluşturma
df = pd.DataFrame({'val1': veri[:, 0], 'val2': veri[:, 1], 'küme': etiketler})

# Küme Görselleştirme
st.write('### Veri Kümeleri')
st.write(df)

st.write('### Veri Görselleştirme')
st.write('Seçilen veri:')
st.write(new_data)

# Seçilen verinin hangi küme içinde olduğunu tahmin etme
tahmin_edilen_küme = kmeans.predict(new_data)[0]
st.write(f'Seçilen veri için tahmini küme: Küme {tahmin_edilen_küme}')

# Görselleştirme
fig, ax = plt.subplots()
scatter = ax.scatter(df['val1'], df['val2'], c=df['küme'], cmap='viridis')
legend = ax.legend(*scatter.legend_elements(), title='Kümeler')
ax.add_artist(legend)
ax.scatter(new_data[:, 0], new_data[:, 1], color='red', marker='X', label='Seçilen Veri')
ax.scatter(merkezler[:, 0], merkezler[:, 1], color='black', marker='o', s=100, label='Küme Merkezleri')
ax.set_xlabel('val1')
ax.set_ylabel('val2')
ax.set_title('K-Meanss ile Veri Kümeleri')
st.pyplot(fig)