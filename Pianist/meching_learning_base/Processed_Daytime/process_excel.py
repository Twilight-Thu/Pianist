import pandas as pd
import os
import shutil

result = []
def excel_one_line_to_list():
    path = r"C:\Users\TWJ\Desktop\meching_learning_base\Daytime\anno\Daytime_train.csv"
    df = pd.read_csv(path, usecols=[0])
    df_list = df.values.tolist()
    for example in df_list:
        result.append(example[0])
    print(result)

base_dir = r"C:\Users\TWJ\Desktop\meching_learning_base\Daytime"
src_images = r"C:\Users\TWJ\Desktop\meching_learning_base\Daytime\images_rotate"
excel_one_line_to_list()
for filename in os.listdir(src_images):
    print(filename)
    if filename in result:
        shutil.copy(os.path.join(base_dir + "\images_rotate\\" + filename), os.path.join(base_dir + "\\train"))

