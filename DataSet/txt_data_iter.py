import random
import cv2
import mxnet as mx
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
def get_label_idx(label):
    unique_label = list(set(label))
    label_idx = []
    for i in unique_label:
        label_idx.append([j for j,x in enumerate(label) if x == i])
    return unique_label,label_idx
class Data_iter():
    def __init__(self, data_names, data_shapes, txt_path,
                 label_names, label_shapes, num_batches=1000,img_mean = []):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.data_gen,self.label = get_data_by_txt(txt_path)
        self.unique_label,self.unique_label_idx = get_label_idx(self.label)
        self.num_batches = num_batches
        self.data_shape = data_shapes[0]
        self.batch_size = data_shapes[0][0]
        self.cur_batch = 0
        self.same_flag = 1
        self.diff_flag = -1
        self.trip_let_list = []
        self.img_mean = img_mean
    def generate_random_sample(self):
        label_num = 4
        rand_idx = random.sample(range(len(self.unique_label)),self.batch_size/label_num)
        idx = []
        for i in rand_idx:
            try:
                idx += random.sample(self.unique_label_idx[i],label_num)
            except:
                for j in range(label_num/2):
                    idx += random.sample(self.unique_label_idx[i], 2)
        return idx

    def generate_data(self):
        data = mx.nd.zeros(self._provide_data[0][1], dtype='float32')
        label = []
        rand_idx = self.generate_random_sample()
        #rand_idx = random.sample(range(len(self.data_gen)),self.batch_size)
        for key, value in enumerate(rand_idx):
            label.append(self.label[value])
            data_name = self.data_gen[value]
            #image = mx.image.imdecode(open(data_name).read())
            image = cv2.imread(data_name)
            image = image[..., ::-1]
            image = mx.nd.array(image)
            image = mx.image.imresize(image, 256, 256)
            if self.data_shape[2] == 224:
                image,_ = mx.image.random_crop(image,(224,224))
            image = image.astype('float32')
            image = image - self.img_mean
            image = mx.nd.swapaxes(image, 0, 2)
            image = mx.nd.swapaxes(image, 1, 2)
            image = image.reshape((1,) + image.shape)
            data[key, :, :, :] = image
        return data,label
    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data,label = self.generate_data()
            return Batch([data],
                         [mx.nd.array(label)])
        else:
            raise StopIteration
if __name__ == '__main__':
    # from train import config as config_class
    # config = config_class.config()
    # total_img_list, cluster, label = get_cub_data('train')
    # data = Data_iter(['data'],[(128,3,256,256)],'train',['label'],[(128,)],num_batches=100,
    #                      img_mean=config.img_mean)
    # for nbatch, data_batch in enumerate(data):
    #     print nbatch
    #     pass
    # data.generate_data()
    pass