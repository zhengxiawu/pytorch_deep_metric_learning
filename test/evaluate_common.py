import torch
import torch.nn as nn
import time
from sklearn.metrics import pairwise_distances
import torchvision.transforms as transforms
from  torch.autograd import Variable

from PIL import Image

import numpy as np

def channel_check(tensor):
    if tensor.shape[0]==1:
        temp = torch.ones((3,tensor.shape[1],tensor.shape[2]))
        temp[0, :, :] = tensor
        temp[1, :, :] = tensor
        temp[2, :, :] = tensor
        return temp
    else:
        return tensor
def get_feature(model,name_list,test_mod,normalize,size,network_dict):
    feature = []
    to_tensor = transforms.ToTensor()
    from tqdm import tqdm
    if test_mod == 'resize':
        scaler = transforms.Scale(size=size)
        count = 0
        for i in tqdm(range(len(name_list))):
            img_path = name_list[i]
            img = Image.open(img_path)
            img_tensor = to_tensor(scaler(img))
            img_tensor = channel_check(img_tensor)
            img_tensor = img_tensor * 255
            t_image = torch.autograd.Variable(normalize(img_tensor).unsqueeze(0)).cuda()
            im_feature = model(t_image,scda=network_dict['scda'],
                               pool_type = network_dict['pool_type'],
                           is_train = False,scale = network_dict['scale'])
            im_feature = im_feature.cpu().detach().numpy()
            feature.append(im_feature)
            count += 1
    else:
        feature = []
        count = 0
        for i in tqdm(range(len(name_list))):
            img_path = name_list[i]
            img = Image.open(img_path)
            w,h = img.size
            if min(h, w) > 700:
                size = (int(round(w * (700. / min(h, w)))), int(round(h * (700. / min(h, w)))))
            else:
                size = (h,w)
            scaler = transforms.Scale(size=size)
            img_tensor = to_tensor(scaler(img))
            img_tensor = channel_check(img_tensor)
            img_tensor = img_tensor * 255
            # test = normalize(img_tensor)
            # test = test.numpy()
            t_image = torch.autograd.Variable(normalize(img_tensor).unsqueeze(0)).cuda()
            im_feature = model(t_image, scda=network_dict['scda'],
                               pool_type=network_dict['pool_type'],
                               is_train=False, scale=network_dict['scale'])
            im_feature = im_feature.cpu().detach().numpy()
            feature.append(im_feature)
            count += 1
    feature = np.array(feature)
    feature = np.reshape(feature, (feature.shape[0], feature.shape[2]))
    return np.array(feature)
def get_info_by_label(label):
    query_id = []
    unique_list = list(set(label))
    retrieval_list = []
    label_dict = {}
    for i in unique_list:
        if i in label_dict:
            continue
        else:
            label_dict.update({i:[idx for idx,x in enumerate(label) if x == i]})
    for key,value in enumerate(label):
        gt = label_dict[value]
        query_id.append(key)
        retrieval_list.append([gt,[key]])
    return query_id,retrieval_list
def get_data_by_txt(txt_path):
    data_gen = []
    label = []
    with open(txt_path) as f:
        txt_lines = f.readlines()
    for key,value in enumerate(txt_lines):
        info_list = value.split(' ')
        data_gen.append(info_list[0])
        label.append(int(info_list[1]))
    return data_gen,label
def get_query_info_by_txt(txt_path):
    data, label = get_data_by_txt(txt_path)
    query_id, retrieval_list = get_info_by_label(label)
    return data,query_id,retrieval_list
def get_model_by_name_and_path(name,path,mode='pytorch'):
    import models
    # in test the num_class end embed_dim can be any number

    #load the parameters
    if mode == 'pytorch':
        model = models.create(name, Embed_dim=512,
                              num_class=100,
                              pretrain=False)
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_dict.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        from models import mxnet_resnet_50
        model = mxnet_resnet_50(pretrain=True,num_class=100)
    return model

def recall_at_k_pipe_line(model,test_mode,txt_path,size,normalize,top_k,network_dict):
    #get recall
    data, query_id, retrieval_list = get_query_info_by_txt(txt_path)
    feature = get_feature(model,data,test_mode,normalize,size,network_dict)
    #np.save(save_feature_name,feature)
    result = recall_at_k(feature,query_id,retrieval_list,top_k)
    return result

def recall_at_k(feature,query_id,retrieval_list,top_k):
    distance = pairwise_distances(feature,feature)
    #distance = compute_distances_self(feature),metric='cosine'
    result = 0
    for i in range(len(query_id)):
        query_distance = distance[query_id[i],:]
        gt_list = retrieval_list[i][0]
        ignore_list = retrieval_list[i][1]
        query_sorted_idx = np.argsort(query_distance)
        query_sorted_idx = query_sorted_idx.tolist()
        result_list = get_result_list(query_sorted_idx,gt_list,ignore_list,top_k)
        #print result_list
        result += 1. if sum(result_list)>0 else 0
    result = result/float(len(query_id))
    return result

def get_result_list(query_sorted_idx,gt_list,ignore_list,top_k):
    return_retrieval_list = []
    count = 0
    while len(return_retrieval_list)<top_k:
        query_idx = query_sorted_idx[count]
        if query_idx in ignore_list:
            pass
        else:
            if query_idx in gt_list:
                return_retrieval_list.append(1)
            else:
                return_retrieval_list.append(0)
        count+=1
    return return_retrieval_list