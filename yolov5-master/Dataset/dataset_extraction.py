import os, shutil
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np

#Name of the folder with annotations
annotation_dir = 'Annotations'

#Name of the folder with the JPG files
jpg_images_dir = 'JPEGImages'

#Name of folder containing train/valid
data_folder = "boatData"

annotations = sorted(glob(os.path.join(os.getcwd(), annotation_dir, '*.xml')))

df = []
cnt = 0
for file in annotations:
    prev_filename = file.split('/')[-1].split('.')[0] + '.jpg'
    filename = str(cnt) + '.jpg'
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        edges = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = int(node.find('bndbox/ymin').text)
        ymax = int(node.find('bndbox/ymax').text)

        row = [prev_filename, filename, edges, xmin, xmax, ymin, ymax]
        df.append(row)
    cnt += 1

data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'edges', 'xmin', 'xmax', 'ymin', 'ymax'])

data[['prev_filename', 'filename', 'edges', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv(
    os.path.join(os.getcwd(),'boat_detection.csv'), index=False)
data.head(10)

img_width = 640
img_height = 480


def width(df):
    return int(df.xmax - df.xmin)


def height(df):
    return int(df.ymax - df.ymin)


def x_center(df):
    return int(df.xmin + (df.width / 2))


def y_center(df):
    return int(df.ymin + (df.height / 2))


def w_norm(df):
    return df / img_width


def h_norm(df):
    return df / img_height


df = pd.read_csv(os.path.join(os.getcwd(),'boat_detection.csv'))

le = preprocessing.LabelEncoder()
le.fit(df['edges'])

print(le.classes_)

labels = le.transform(df['edges'])
df['labels'] = labels

df['width'] = df.apply(width, axis=1)
df['height'] = df.apply(height, axis=1)

df['x_center'] = df.apply(x_center, axis=1)
df['y_center'] = df.apply(y_center, axis=1)

df['x_center_norm'] = df['x_center'].apply(w_norm)
df['width_norm'] = df['width'].apply(w_norm)

df['y_center_norm'] = df['y_center'].apply(h_norm)
df['height_norm'] = df['height'].apply(h_norm)

df.head(30)


df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)

print(df_train.shape, df_valid.shape)

mk_dir_path = os.path.join(os.getcwd(),data_folder)
all_images_path = os.path.join(mk_dir_path, 'images')
train_images_path = os.path.join(mk_dir_path, 'images','train');
valid_images_path = os.path.join(mk_dir_path, 'images','valid')

all_labels_path = os.path.join(mk_dir_path, 'labels')
train_labels_path = os.path.join(mk_dir_path, 'labels','train');
valid_labels_path = os.path.join(mk_dir_path, 'labels','valid')


os.mkdir(mk_dir_path)
os.mkdir(all_images_path)
os.mkdir(train_images_path)
os.mkdir(valid_images_path)

os.mkdir(all_labels_path)
os.mkdir(train_labels_path)
os.mkdir(valid_labels_path)

def segregate_data(df, img_path, label_path, train_img_path, train_label_path):
    filenames = []
    for filename in df.filename:
        filenames.append(filename)
    filenames = set(filenames)

    for filename in filenames:
        yolo_list = []

        for _,row in df[df.filename == filename].iterrows():
            yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

        yolo_list = np.array(yolo_list)
        txt_filename = os.path.join(train_label_path,str(row.prev_filename.split('\\')[-1]).split('.')[0]+'.txt' )

        np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(os.path.join(img_path, str(row.prev_filename.split('\\')[-1])), os.path.join(train_img_path, str(row.prev_filename.split('\\')[-1])))

## Apply function ##

src_path= os.getcwd()
src_img_path = os.path.join(src_path, jpg_images_dir)
src_label_path = os.path.join(src_path, annotation_dir)

segregate_data(df_train, src_img_path, src_label_path, train_images_path, train_labels_path)
segregate_data(df_valid, src_img_path, src_label_path, valid_images_path, valid_labels_path)

print("No. of Training images", len(os.listdir(train_images_path)))
print("No. of Training labels", len(os.listdir(train_labels_path)))

print("No. of valid images", len(os.listdir(valid_images_path)))
print("No. of valid labels", len(os.listdir(valid_labels_path)))