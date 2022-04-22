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

#ファイルを読み込む
# Use pd.read_csv to read file
path = "Star99999_raw.csv"
raw_data = pd.read_csv(path)

raw_data

#項目を全て表示する
#display the feature
raw_data.columns
print(raw_data.columns)

#一覧を表示
#display all
raw_data.describe()
print(raw_data.describe)

#データの要約
#the sammary
#the column = カラム名、Non-Null Count = データのある行数、Dtype = データの型
raw_data.info()

#全てをfloat型へ変換する
#convert Columns data type to float values
raw_data["Vmag"] = pd.to_numeric(raw_data["Vmag"], downcast="float", errors='coerce')
raw_data["Plx"] = pd.to_numeric(raw_data["Plx"], downcast="float", errors='coerce')
raw_data["e_Plx"] = pd.to_numeric(raw_data["e_Plx"], downcast="float", errors='coerce')
raw_data["B-V"] = pd.to_numeric(raw_data["B-V"], downcast="float", errors='coerce')

#中身を確認
#check the DataType of our dataset
raw_data.info()

#最大値や平均値などの数値を表示させる
# Actually , if you want to show all the columns you can add parameter `include='all'`.
raw_data.describe(include='all')
print(raw_data.describe(include='all'))


#以下より、欠陥のあるデータの削除を行う
#following the program to delete missing the data

#欠損値データの個数を確認する
#check the totalnumber of the missing data by '.isnull().sum()'
missing_values_count = raw_data.isnull().sum()
print(missing_values_count)

#全体のうち、欠損データは何パーセントか確認する
#how many total missing values do we have?
#.shapeで配列の形状を確認
#check the array shape by '.shape'
#.formatで置換を行っている
#'.format' is a replacement field
total_cells = np.product(raw_data.shape)
total_missing = missing_values_count.sum()
missing_values_count

percent_missing = (total_missing/total_cells)
print("Percentage Missing:", "{:.3%}".format(percent_missing))

#.dropnaで欠損データを削除
#delete the missing data by '.dropna'
raw_data_na_dropped = raw_data.dropna() 

#削除するデータ数はいくらか
#just how much rows did we drop?
dropped_rows_count = raw_data.shape[0]-raw_data_na_dropped.shape[0]
print("Rows we dropped from original dataset: %d \n" % dropped_rows_count)
#全体の何パーセント分を削除するのか
#percentage we dropped
percent_dropped = dropped_rows_count/raw_data.shape[0]
print("Percentage Loss:", "{:.2%}".format(percent_dropped))

#再度、中身を確認
#check the data again
raw_data_na_dropped.describe()
print(raw_data_na_dropped.describe())

#Columnを1つ消して、新しいColumnを作る
#delete one column and make new column

#unnamedを読み取らせないようにする
#prevent 'unnamed' from being read
raw_data_na_dropped = raw_data_na_dropped.drop('Unnamed: 0', axis=1)
raw_data_na_dropped

#本当にunnamedが消せたかどうか確認する
#check not to read 'unnamed'
raw_data_na_dropped.describe()
print(raw_data_na_dropped.describe())

raw_data_na_dropped.info()
print(raw_data_na_dropped.info())

#'drop=True'で削除したいデータの行を一括削除する
#drop completely the line of the data that we wnat to delete
#'.reset_index' で削除された後のデータを上から番号をふり直す（0から）
#renumber the new data from the top by '.reset_index' (from 0)
raw_data_na_dropped_reindex = raw_data_na_dropped.reset_index(drop=True)

#本当にデータを削除できたか確認
#check the data was dropped
raw_data_na_dropped_reindex.info()
print(raw_data_na_dropped_reindex.info())
#Optional - Save our progress
#raw_data_na_dropped_reindex.to_csv("Star99999_na_dropped.csv", index=False)

df = raw_data_na_dropped_reindex.copy()
df

#Plxが0であるものを、排除する
#Dropping rows that `Plx` = 0
df = df[df.Plx != 0]
df = df.reset_index(drop=True)
df
print(df)


#新しいColumnである'Amag'を作る
#make new column 'Amag'
df["Amag"] = df["Vmag"] + 5* (np.log10(abs(df["Plx"]))+1)
df

print(df)

df['SpType']

#Target ClassにSptypeをコピーする
#Copy the SpType column to a new column called TargetClass
df['TargetClass'] = df['SpType']
df
print(df)

#Dwarfs (I, II, III, VII),Giants (IV, V, VI),Other Special Stars (None)

#Target Classを決める
#create Target Class

#The intuitive approach (Could take a long time if you have a huge dataset)

for i in range(len(df['TargetClass'])):
    if "V" in df.loc[i,'TargetClass']: 
        if "VII" in df.loc[i,'TargetClass']: 
            df.loc[i,'TargetClass'] = 0 # VII is Dwarf
        else:
            df.loc[i,'TargetClass'] = 1 # IV, V, VI are Giants
    elif "I" in df.loc[i,'TargetClass']: 
        df.loc[i,'TargetClass'] = 0 # I, II, III are Dwarfs
    else: 
        df.loc[i,'TargetClass'] = 9 # None
        
df['TargetClass']

print(df['TargetClass'])

#check all
df.describe(include='all')

#以下より、データ数の調整を行う
#By following code, I will balance the volume of data

#check the volume of 0data,1data and 9data
df['TargetClass'].value_counts()

#Dropped data9
df = df[df.TargetClass != 9]

#by using ".reset_index", number from 0 in a sequence
df = df.reset_index(drop=True)

df
print(df)

#make Labels
df_giants = df[df.TargetClass == 1]
df_dwarfs = df[df.TargetClass == 0]

#check that labels were got correctly
# Numbers of rows of Giants and Dwarfs
num_of_giant = df_giants.shape[0]
num_of_dwarf = df_dwarfs.shape[0]
print("Giants(1):",num_of_giant)
print("Dwarfs(0):",num_of_dwarf)

from sklearn.utils import resample

#Downsample majority class
df_giants_downsampled = resample(df_giants,replace=False,n_samples=num_of_dwarf,random_state=1)
df_downsampled = pd.concat([df_giants_downsampled, df_dwarfs])

#combine minority class with downsampled majority class by'pd.contact'
df_downsampled = pd.concat([df_giants_downsampled, df_dwarfs])

#check that the volume of 0data is the same as that of 1data
Balance_data = df_downsampled['TargetClass'].value_counts()
print(Balance_data)

#final confirmation
df_balanced = df_downsampled.reset_index(drop=True)

df_balanced.info()

df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
print(df_balanced)

#Save our dataset, we can finally play with it!!!
df_balanced.to_csv("Star39552_balanced.csv", index=False)