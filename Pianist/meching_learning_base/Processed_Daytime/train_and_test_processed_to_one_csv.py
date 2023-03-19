import os
import pandas as pd
import numpy as np

path_train = r"C:\Users\TWJ\Desktop\meching_learning_base\Processed_Daytime\anno\Daytime_train.csv"
path_test = r"C:\Users\TWJ\Desktop\meching_learning_base\Processed_Daytime\anno\Daytime_test.csv"

df_train = pd.read_csv(path_train, header=0)
df_test = pd.read_csv(path_test, header=0)

# print(df_train)
# print("------------------")
# print(df_test)

path_store = r"./merge.csv"
df_train = df_train[["fileId", "PM2.5"]]
df_test = df_test[["fileId", "PM2.5"]]
df_train.to_csv(path_store, mode='a+')
df_test.to_csv(path_store, mode='a+')
