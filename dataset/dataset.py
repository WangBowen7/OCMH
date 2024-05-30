import h5py
from PIL import Image
import numpy as np


import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# import settings



def load_dataset(dataname='MIRFlickr', split_ratio=37, train_mode=None, test_mode=None, dataset_mode=None):
    global all_data, split_index, I_tr, I_te, I_db, T_tr, T_te, T_db, L_tr, L_te, L_db


    if dataname == 'MIRFlickr':
        flickr_path = '/data/WangBoWen/OpenSet/feature/openset{}/MIRFlickr_openset.mat'.format(split_ratio)
        all_data = h5py.File(flickr_path)

        flickr_index_path = '/data/WangBoWen/OpenSet/split/openset{}/MIRFlickr_index.mat'.format(split_ratio)
        split_index = h5py.File(flickr_index_path)

        pass

    if dataname == 'NUSWIDE':
        nuswide_path = '/data/WangBoWen/OpenSet/feature/openset{}/NUSWIDE_openset.mat'.format(split_ratio)
        all_data = h5py.File(nuswide_path)

        # nuswide_index_path = '/data/WangBoWen/OpenSet/split/openset{}/NUSWIDE_index.mat'.format(split_ratio)
        nuswide_index_path = '/data/WangBoWen/OpenSet/split/openset{}/NUSWIDE_index_new.mat'.format(split_ratio)
        split_index = h5py.File(nuswide_index_path)
        pass

    if dataname == 'MSCOCO':
        coco_path = '/data/WangBoWen/OpenSet/feature/openset{}/MSCOCO_openset.mat'.format(split_ratio)
        all_data = h5py.File(coco_path)

        # coco_index_path = '/data/WangBoWen/OpenSet/split/openset{}/MSCOCO_index.mat'.format(split_ratio)
        coco_index_path = '/data/WangBoWen/OpenSet/split/openset{}/MSCOCO_index_new.mat'.format(split_ratio)
        split_index = h5py.File(coco_index_path)

    source_image = all_data['source_image'][:].T
    source_text = all_data['source_text'][:].T
    source_label = all_data['source_label'][:].T

    target_image = all_data['target_image'][:].T
    target_text = all_data['target_text'][:].T
    target_label = all_data['target_label'][:].T

    if train_mode == 'source':
        source_train_index = np.squeeze(split_index['source_train_index'][:])
        I_tr = source_image[source_train_index]
        T_tr = source_text[source_train_index]
        L_tr = source_label[source_train_index]
    elif train_mode == 'target':
        target_train_index = np.squeeze(split_index['target_train_index'][:])
        I_tr = target_image[target_train_index]
        T_tr = target_text[target_train_index]
        L_tr = target_label[target_train_index]

    if test_mode == 'source':

        source_test_index = np.squeeze(split_index['source_test_index'][:])
        I_te = source_image[source_test_index]
        T_te = source_text[source_test_index]
        L_te = source_label[source_test_index]

    elif test_mode == 'target':

        target_test_index = np.squeeze(split_index['target_test_index'][:])
        I_te = target_image[target_test_index]
        T_te = target_text[target_test_index]
        L_te = target_label[target_test_index]

    elif test_mode == 'target_only':
        if dataname == 'NUSWIDE' or dataname == 'MSCOCO':
            target_only_test_index = np.squeeze(split_index['target_only_test_index'][:])
            I_te = target_image[target_only_test_index]
            T_te = target_text[target_only_test_index]
            L_te = target_label[target_only_test_index]
        else:
            print('dataset error, no target_only index')


    if dataset_mode == 'source':

        source_retrieval_index = np.squeeze(split_index['source_retrieval_index'][:])
        I_db = source_image[source_retrieval_index]
        T_db = source_text[source_retrieval_index]
        L_db = source_label[source_retrieval_index]

    elif dataset_mode == 'target':

        target_retrieval_index = np.squeeze(split_index['target_retrieval_index'][:])
        I_db = target_image[target_retrieval_index]
        T_db = target_text[target_retrieval_index]
        L_db = target_label[target_retrieval_index]

    elif dataset_mode == 'mix':

        source_retrieval_index = np.squeeze(split_index['source_retrieval_index'][:])
        target_retrieval_index = np.squeeze(split_index['target_retrieval_index'][:])

        I_db = np.vstack((source_image[source_retrieval_index], target_image[target_retrieval_index]))
        T_db = np.vstack((source_text[source_retrieval_index], target_text[target_retrieval_index]))
        L_db = np.vstack((source_label[source_retrieval_index], target_label[target_retrieval_index]))

    elif test_mode == 'target_only':
        if dataname == 'NUSWIDE' or dataname == 'MSCOCO':
            target_only_retrieval_index = np.squeeze(split_index['target_only_retrieval_index'][:])
            I_db = target_image[target_only_retrieval_index]
            T_db = target_text[target_only_retrieval_index]
            L_db = target_label[target_only_retrieval_index]
        else:
            print('dataset error, no target_only index')

    # 只加载训练数据
    if test_mode == None and dataset_mode == None and train_mode !=None:
        return I_tr, T_tr, L_tr
    # 只加载测试数据
    elif train_mode == None and (test_mode !=None and dataset_mode!=None):
        if dataset_mode == 'mix':
            return I_te, I_db, T_te, T_db, L_te, L_db, source_retrieval_index, target_retrieval_index
        else:
            return I_te, I_db, T_te, T_db, L_te, L_db
    elif test_mode != None and (dataset_mode == None and train_mode ==None):
        return I_te, T_te, L_te
    # 加载训练数据+测试数据
    else:
        return I_tr, I_te, I_db, T_tr, T_te, T_db, L_tr, L_te, L_db


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count


class TrainDataSet(Dataset):
    def __init__(self, images, texts, image_labels, text_labels):
        self.images = images
        self.texts = texts
        self.image_labels = image_labels
        self.text_labels = text_labels

    def __getitem__(self, index):
        image = self.images[index]
        text = self.texts[index]

        image_label = self.image_labels[index]
        text_label = self.image_labels[index]
        return image, text, image_label, text_label, index

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.image_labels)
        assert len(self.texts) == len(self.text_labels)
        return count

def get_dataloader():

    pass

if __name__ == '__main__':

    pass