import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow # develop and train models


from sklearn.model_selection import train_test_split

import os
import shutil


# shutil.rmtree('/kaggle/working/data') # deletes the directory and all its contents

# [CREATE THE BASE DIRECTORY]

# data
base_directory = 'data'
os.mkdir(base_directory)

# [CREATE THE TRAIN AND VALIDATION DIRECTORIES]

# train
train_directory = os.path.join(base_directory, 'train')
os.mkdir(train_directory)

# val_directory
validation_directory = os.path.join(base_directory, 'val')
os.mkdir(validation_directory)

# [CREATE FOLDERS INSIDE THE TRAIN AND VALIDATION DIRECTORIES]

# create folders inside train
nv = os.path.join(train_directory, 'nv')
os.mkdir(nv)
mel = os.path.join(train_directory, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_directory, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_directory, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train_directory, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train_directory, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_directory, 'df')
os.mkdir(df)

# create folders inside val_directory
nv = os.path.join(validation_directory, 'nv')
os.mkdir(nv)
mel = os.path.join(validation_directory, 'mel')
os.mkdir(mel)
bkl = os.path.join(validation_directory, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(validation_directory, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(validation_directory, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(validation_directory, 'vasc')
os.mkdir(vasc)
df = os.path.join(validation_directory, 'df')
os.mkdir(df)

df_data = pd.read_csv('dataverse_files/HAM10000_metadata') # dataframe
df_data.head()


# this will tell me how many images are associated with each lesion_id
df = df_data.groupby('lesion_id').count()

# now I filter out lesion_id's that have only one image associated with it
df = df[df['image_id'] == 1]

df.reset_index(inplace=True)

df.head()


# here I identify lesion_id's that have duplicate images and those that have only
# one image.

def identify_duplicates(x):
    unique_list = list(df['lesion_id'])
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'


# create a new colum that is a copy of the lesion_id column
df_data['duplicates'] = df_data['lesion_id']
# apply the function to this new column
df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)

df_data.head()

df_data['duplicates'].value_counts()

df = df_data[df_data['duplicates'] == 'no_duplicates']

df.shape

# create the validation set
y = df['dx']

_, df_validation = train_test_split(df, test_size=0.17, random_state=101, stratify=y)

df_validation.shape


# This set will be df_data excluding all rows that are in the Validation Set

# This function identifies if an image is part of the train or val set.
def identify_validation_rows(x):
    # create a list of all the lesion_id's in the val set
    validation_list = list(df_validation['image_id'])
    if str(x) in validation_list:
        return 'validation'
    else:
        return 'train'


# identify train and val rows

# create a new colum that is a copy of the image_id column
df_data['train_or_validation'] = df_data['image_id']
# apply the function to this new column
df_data['train_or_validation'] = df_data['train_or_validation'].apply(identify_validation_rows)

# filter out train rows
df_train = df_data[df_data['train_or_validation'] == 'train']

print("Number of images in the Train Set = ", len(df_train))
df_train['dx'].value_counts()

print("Number of images in the Validation Set = ", len(df_validation))
df_validation['dx'].value_counts()

# get the image_id as the index in df_data
df_data.set_index('image_id', inplace=True)

# get a list of images in each of the two folders
images_folder_1 = os.listdir('dataverse_files/HAM10000_images_part_1')
images_folder_2 = os.listdir('dataverse_files/HAM10000_images_part_2')

# get a list of Train and Validation images
train_images_list = list(df_train['image_id'])
validation_images_list = list(df_validation['image_id'])

# transfer the Train Set images

for image in train_images_list:

    fname = image + '.jpg'
    label = df_data.loc[image, 'dx']

    if fname in images_folder_1:
        # source path to image
        source_path = os.path.join('dataverse_files/HAM10000_images_part_1', fname)
        # destination path to image
        destination_path = os.path.join(train_directory, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(source_path, destination_path)

    if fname in images_folder_2:
        # source path to image
        source_path = os.path.join('dataverse_files/HAM10000_images_part_2', fname)
        # destination path to image
        destination_path = os.path.join(train_directory, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(source_path, destination_path)

# transfer the Validation Set images

for image in validation_images_list:

    fname = image + '.jpg'
    label = df_data.loc[image, 'dx']

    if fname in images_folder_1:
        # source path to image
        source_path = os.path.join('dataverse_files/HAM10000_images_part_1', fname)
        # destination path to image
        destination_path = os.path.join(validation_directory, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(source_path, destination_path)

    if fname in images_folder_2:
        # source path to image
        source_path = os.path.join('dataverse_files/HAM10000_images_part_2', fname)
        # destination path to image
        destination_path = os.path.join(validation_directory, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(source_path, destination_path)


print("nv    : ", len(os.listdir('data/train/nv')))
print("mel   : ", len(os.listdir('data/train/mel')))
print("bkl   : ", len(os.listdir('data/train/bkl')))
print("bcc   : ", len(os.listdir('data/train/bcc')))
print("akiec : ", len(os.listdir('data/train/akiec')))
print("vasc  : ", len(os.listdir('data/train/vasc')))
print("df    : ", len(os.listdir('data/train/df')))


print("nv    : ", len(os.listdir('data/val/nv')))
print("mel   : ", len(os.listdir('data/val/mel')))
print("bkl   : ", len(os.listdir('data/val/bkl')))
print("bcc   : ", len(os.listdir('data/val/bcc')))
print("akiec : ", len(os.listdir('data/val/akiec')))
print("vasc  : ", len(os.listdir('data/val/vasc')))
print("df    : ", len(os.listdir('data/val/df')))


