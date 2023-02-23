import pandas as pd
import matplotlib.pyplot as plt

url = "http://bilkav.com/satislar.csv"
veriler = pd.read_csv(url)
#print(veriler)
aylar = veriler[['Aylar']] #bağımsız değişken
satislar = veriler[['Satislar']] #bağımlı değişken

#verileri test ve eğitim(train) olarak ayıralım
from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(aylar,satislar,test_size=0.33 ,random_state=0)

#verileri ölçeklendirelim
"""from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train =sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)"""

#linear regression yapalım
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train ,y_train)
#tahmin edelim
tahmin = lr.predict(x_test)

#çıkan sonuçları sıralayalım
x_train =x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")