import evaluate_common
import os
import torchvision.transforms as transforms
if __name__ == '__main__':
    root_dir = '/home/zhengxiawu/project/pytorch_deep_metric_learning'
    model_path = os.path.join(root_dir,'checkpoints/cub_m_01_1e4_n_4_b_64/20_model.pth')
    load_num_round = 0
    data = 'cub'
    conv_type = 'resnet_50'
    size = (256,256)
    top_k = 1
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_mode = 'resize'
    if data == 'cub':
        txt_path = os.path.join(root_dir,'data_txt/cub_test_resource.txt')
    model = evaluate_common.get_model_by_name_and_path(conv_type,model_path)
    model.cuda()
    model.eval()
    network_dict = {}
    network_dict['scda'] = False
    network_dict['pool_type'] = 'max_avg'
    network_dict['scale'] = 128
    result = evaluate_common.recall_at_k_pipe_line(model, test_mode, txt_path,
                                                   size, normalize, 1, network_dict)
    print result
    # get_network