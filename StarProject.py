#必要なライブラリをインストールする
#install the libraries

#numpy(計算処理のライブラリ)のインストール
#install numpy(the library for calculation processing)
import numpy as np

#pandas(データ解析にライブラリ)のインストール
#installed pandas(the library for data analysis)
import pandas as pd

#osを呼び出し、ファイルを読み込む準備をする
#call os ,and prepare to read a file
#dirnameでファイルのパスを取得する
#get file pass by 'dirname'
#os.walkで（ファイルの中のファイルなど）ファイルの中身を表示する
#display the list of the file by 'os.walk'
import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#install matplot and seaborn to make graphs
import matplotlib.pyplot as plt
import seaborn as sns

#apply the basic style of seaborn
sns.set()
import statsmodels.api as sm

#read the balnce data
raw_df = pd.read_csv("Star39552_balanced.csv")
raw_df

print(raw_df)
print("")
print("")

#以下よりFeaturesを選択する
#select features
df = raw_df[['B-V', 'Amag', 'TargetClass']]
df

# Select Target
y = df['TargetClass']

# Select Features
x = df[['B-V','Amag']]

# Splitting the data into train dataset and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=0)

#データの標準化
#Data normalization
#標準化とは「データのばらつきの違いによってデータの重要性を変化させてしまう、ということを防ぐ」こと。
#Normalization is that the importance of the data isn't changed by the data variation
#標準化　＝　(元の値 - 平均) / 標準偏差
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#fitting and transform
#'fitting' is 曲線のあてはめ
#'transform' is to run fitting
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Use logistic regression
from sklearn.linear_model import LogisticRegression
star_predictor = LogisticRegression(random_state=0)

# Start model training
star_predictor.fit(x_train, y_train)

#係数と切片の算出
#caluculate regression coefficient and intercept
star_predictor.coef_
print("regression coefficient is")
print(star_predictor.coef_)
print("")

star_predictor.intercept_
print("intercept is")
print(star_predictor.intercept_)
print("")
print("")


#".score" is 決定係数（そのモデルと精度を表す指標。0~1で表され、1に近いほど良い）
print('the score on train dataset is') 
print(star_predictor.score(x_train, y_train))
print("")

y_pred = star_predictor.predict(x_test)
y_pred_front = y_pred[0:5]
y_pred_last = y_pred[-6:-1]
y_test_front= y_test[0:5]
y_test_last = y_test[-6:-1]
print("Prediction(front):",y_pred_front)
print("Actual(front):")
print("")
print(y_test_front)
print("")

print("Prediction(last):",y_pred_last)
print("Actual(last):",y_pred_last)
print("")

# Confusion Matrix
#                 Predicted
#                 Negative  Positive
#Actual Negative     TN        FP
#       Positive     FN        TP
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n", cm)
print("")
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig('sklearn_confusion_matrix_annot_blues.png')

# Model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))
print("")

