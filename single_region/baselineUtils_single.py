import copy
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam=10000, p=0.0001, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def generate_mean_std():
    data_path = 'PIE_dataset'
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                   'val_data': True,
                                   'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}
    imdb = PIE(data_path=data_path)
    beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    tracks=[]
    tracks_spd=[]
    for track in beh_seq_train['bbox']:
        tracks.extend([track[i:i + 60] for i in
                       range(0, len(track) - 60 + 1, 7)])
    for track in beh_seq_train['obd_speed']:
        tracks_spd.extend([track[i:i + 60] for i in
                       range(0, len(track) - 60 + 1, 7)])
    trac = np.array(tracks).reshape(-1, 4)
    mean = trac.mean(0)
    std = trac.std(0)
    trac1 = np.array(tracks_spd).reshape(-1, 1)
    mean_speed = trac1.mean(0)
    std_speed = trac1.std(0)
    return mean,std,mean_speed,std_speed

global mean
global std



def create_pie_dataset(flag,device,mean=None,std=None):
    if flag=='train':
        label_list=[]
        pos_data_dir= '../data/AD_processed_single/mose2/train/positive/hippocampus'
        neg_data_dir= '../data/AD_processed_single/mose2/train/negative/hippocampus'
        data_list=get_data(pos_data_dir,neg_data_dir)
        data_array=np.concatenate([x[0] for x in data_list],axis=0)

        print(data_array.shape)
        for x in data_list:
            label_list.extend(x[1])
        mean_value=data_array.mean(0)
        std_value=data_array.std(0)
        data = [[np.expand_dims(data_array[i], axis=0),label_list[i]] for i in range(data_array.shape[0])]
        return OnboardTfDataset(data, device), mean_value, std_value
    elif flag=='test':
        label_list=[]
        pos_data_dir = '../data/AD_processed_single/mose2/test/positive/hippocampus'
        neg_data_dir = '../data/AD_processed_single/mose2/test/negative/hippocampus'
        data_list = get_data(pos_data_dir, neg_data_dir)
        data_array = np.concatenate([x[0] for x in data_list], axis=0)
        for x in data_list:
            label_list.extend(x[1])
        data = [[np.expand_dims(data_array[i], axis=0),label_list[i]] for i in range(data_array.shape[0])]
        return OnboardTfDataset(data, device), None, None


def get_tracks(dataset, data_types, observe_length,dataset_type, predict_length, overlap, normalize,mean,std,mean_speed,std_speed):
    """
    Generates tracks by sampling from pedestrian sequences
    :param dataset: The raw data passed to the method
    :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
    JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
    :param observe_length: The length of the observation (i.e. time steps of the encoder)
    :param predict_length: The length of the prediction (i.e. time steps of the decoder)
    :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
    :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
    the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
    :return: A dictinary containing sampled tracks for each data modality
    """
    seq_length = observe_length + predict_length
    overlap_stride = observe_length if overlap == 0 else \
        int((1 - overlap) * observe_length)
    overlap_stride = 1 if overlap_stride < 1 else overlap_stride
    d = {}

    for dt in data_types:
        print('data_type',dt)
        try:
            d[dt] = dataset[dt]
        except KeyError:
            raise ('Wrong data type is selected %s' % dt)

    d['image'] = dataset['image']
    d['pid'] = dataset['pid']
    for k in d.keys():
        tracks = []
        for track in d[k]:
            tracks.extend([track[i:i + seq_length] for i in
                           range(0, len(track) - seq_length + 1, overlap_stride)])
        d[k] = tracks
    print('data_types',data_types)
    if dataset_type=='train' and 'bbox' in data_types:
        trac=np.array(d['bbox']).reshape(-1, 4)
        mean = trac.mean(0)
        std = trac.std(0)
    if dataset_type=='train' and 'obd_speed' in data_types:
        trac1 = np.array(d['obd_speed']).reshape(-1, 1)
        mean_speed=trac1.mean(0)
        std_speed=trac1.std(0)
    if 'bbox' in data_types:
        box=copy.deepcopy(d['bbox'])
    else:
        box=None
    d['scale']=[]

    if normalize:
        if 'bbox' in data_types:
            for i in range(len(d['bbox'])):
                d['bbox'][i] = np.divide(np.subtract(d['bbox'][i], mean),std)
        if 'obd_speed' in data_types:
            for i in range(len(d['obd_speed'])):
                d['obd_speed'][i] = np.divide(np.subtract(d['obd_speed'][i], mean_speed),std_speed).tolist()
        if 'center' in data_types:
            for i in range(len(d['center'])):
                d['center'][i] = np.subtract(d['center'][i], d['center'][i][0]).tolist()
        for k in d.keys():
            if k != 'bbox' and k != 'center'and k!='scale' and k!='obd_speed' and k!='ego_op_flow' and k!='ped_op_flow':
                for i in range(len(d[k])):
                    d[k][i] = d[k][i]

    return d,box

def get_data_helper(data, data_type):
    """
    A helper function for data generation that combines different data types into a single representation
    :param data: A dictionary of different data types
    :param data_type: The data types defined for encoder and decoder input/output
    :return: A unified data representation as a list
    """
    if not data_type:
        return []
    d = []
    for dt in data_type:
        if dt == 'image':
            continue
        d.append(np.array(data[dt]))
    if len(d) > 1:
        for i in d:
            print('sss',i.shape)
        return np.concatenate(d, axis=2)
    else:
        return d[0]

def get_data(pos_dir,neg_dir):
    """
    Main data generation function for training/testing
    :param data: The raw data
    :param model_opts: Control parameters for data generation characteristics (see below for default values)
    :return: A dictionary containing training and testing data
    """
    data_list=[]
    for file in os.listdir(pos_dir):
        data_list.append([np.load(os.path.join(pos_dir,file)),[1]])
    print(len(data_list))
    x=len(data_list)
    for file in os.listdir(neg_dir):
        data_list.append([np.load(os.path.join(neg_dir,file)),[0]])
    print(len(data_list)-x)
    random.shuffle(data_list)
    return data_list


class OnboardTfDataset(Dataset):
    def __init__(self,data,device):
        super(OnboardTfDataset,self).__init__()

        self.data=data
        self.device=device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'input': torch.Tensor([self.data[index][0][0:1,:]]).to(self.device),
                'label': torch.Tensor([self.data[index][1]]).long().to(self.device),
                }


def create_folders(baseFolder,datasetName):
    try:
        os.mkdir(baseFolder)
    except:
        pass

    try:
        os.mkdir(os.path.join(baseFolder,datasetName))
    except:
        pass
