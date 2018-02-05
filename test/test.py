import evaluate_common
import os
import torchvision.transforms as transforms
import torch
if __name__ == '__main__':
    root_dir = '/home/zhengxiawu/project/pytorch_deep_metric_learning'
    model_path = os.path.join(root_dir,'pretrained_models/kit_pytorch.npy')
    model_path = '/home/zhengxiawu/project/pytorch_deep_metric_learning/checkpoints/mxnet_resnet_50_cub_s_1/999_model.pth'
    load_num_round = 0
    data = 'cub'
    network_name = 'mxnet_resnet_50'
    size = (256,256)
    top_k = 1
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #normalize = transforms.Normalize(mean=[123,117,104],
    #                                 std=[1,1,1])
    test_mode = 'resize'
    if data == 'cub':
        txt_path = os.path.join(root_dir,'data_txt/cub_test_resource.txt')
    #model = evaluate_common.get_model_by_name_and_path(network_name,model_path,mode='mxnet')
    model = torch.load(model_path)
    model.cuda()
    model.eval()
    network_dict = {}
    network_dict['scda'] = True
    network_dict['pool_type'] = 'max_avg'
    network_dict['scale'] = 128
    result = evaluate_common.recall_at_k_pipe_line(model, test_mode, txt_path,
                                                   size, normalize, 1, network_dict)
    print result
    # get_network