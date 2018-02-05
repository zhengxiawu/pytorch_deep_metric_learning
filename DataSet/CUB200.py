import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class CUB200:
    def __init__(self, root, train=True, test=True, transform=None):
        # Data loading code
        if transform is None:

            transform = [transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225]),
                transforms.Normalize(mean=[123, 117, 104], std=[1, 1, 1])
            ]),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])]

        if root is None:
            root = '/home/zhengxiawu/data/CUB_200_2011/'

        traindir = os.path.join(root, 'train_images')
        testdir = os.path.join(root, 'test_images')
        if train:
            self.train = datasets.ImageFolder(traindir, transform[0])
        if test:
            self.test = datasets.ImageFolder(testdir, transform[1])
