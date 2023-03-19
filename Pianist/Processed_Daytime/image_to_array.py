from PIL import Image
import numpy as np
import os
import pandas as pd
import torchvision.transforms as transforms

image_dir = "./images_rotate/"
result_dir = "./results/"
all_data = np.empty((0, 3072), float)
images_names = []

def process_input(image):
  image = transforms.Resize((32, 32))(image)
  return image

def image_to_array_file():
  global all_data, images_names
  filenames = os.listdir(image_dir)
  images_array = np.array([])
  for filename in filenames:
    images_names += [filename]
    image = Image.open(image_dir + filename)
    image = process_input(image)
    r, g, b = image.split()
    r_arr = np.array(r).reshape(1, 32 * 32)
    g_arr = np.array(g).reshape(1, 32 * 32)
    b_arr = np.array(b).reshape(1, 32 * 32)
    final_arr = np.concatenate((r_arr, g_arr, b_arr), axis=1)
    print(final_arr.shape)
    all_data = np.vstack((all_data, final_arr))


image_to_array_file()
print("Finished process!")
data = pd.DataFrame(all_data, index=images_names)
data.to_excel(os.path.join(result_dir, "results_final.xlsx"))