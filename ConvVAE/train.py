import random
import numpy as np

import pandas as pd
import glob
import os
import imageio
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from pytorch_model_summary import summary

from model import VariationalAutoencoder
from train_aux import train_vae

def fetch_dataset(dx=50, dy=50, dimx=128, dimy=128):
    
    df_attrs = pd.read_csv(ATTRIBUTES_PATH, sep='\t', skiprows=1,) 
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])
    
    photo_ids = []
    for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})

    photo_ids = pd.DataFrame(photo_ids)
    df = pd.merge(df_attrs,photo_ids,on=('person','imagenum'))

    assert len(df)==len(df_attrs),"lost some data when merging dataframes"
    
    all_photos = df['photo_path'].apply(imageio.imread)\
                                .apply(lambda img:img[dy:-dy,dx:-dx])\
                                .apply(lambda img: np.array(Image.fromarray(img).resize([dimx,dimy])) )

    all_photos = np.stack(all_photos.values).astype('uint8')
    all_attrs = df.drop(["photo_path","person","imagenum"],axis=1)
    
    return all_photos,all_attrs


ATTRIBUTES_PATH = "lfw_attributes.txt"
DATASET_PATH ="lfw-deepfunneled/lfw-deepfunneled/"

dataset = []
for path in glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg")):
    person = path.split("/")[-2]
    dataset.append({"person":person, "path": path})
    
dataset = pd.DataFrame(dataset)
dataset = dataset.groupby("person").filter(lambda x: len(x) < 25 )
dataset.head(5)

cut = 50 # Remove pixels of background from the sides

SHAPE = 128 #Must be multiple of 16
LATENT_DIMS = 64
SH = SHAPE // 8 # Used to construct the VAE

print('Fetching Dataset')
data, attrs = fetch_dataset(dx = cut, dy = cut, dimx = SHAPE, dimy = SHAPE)
data = np.array(data / 255, dtype='float32').transpose((0, 3, 1, 2))

device = 'cuda:0'

print('Loading Model')
model = VariationalAutoencoder(latents = LATENT_DIMS, c = 4).to(device)
print(summary(model, torch.zeros((1, *data.shape[1:])).to(device), show_input=True))

history_vae = train_vae(data, model.to(device), epochs=400, batch_size=4, lr=1e-4)

textfile = open("history.txt", "+w")
for element in history_vae:
    textfile.write(str(element) + "\n")
textfile.close()