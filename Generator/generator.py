import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torchvision
from PIL import Image
import torch.utils.data as data_utils
import argparse

NumWorkers = 4


def get_dataset_discrete(dataset, group=-1, path=None, args=None, res_selector=None, transforms=None,
                         flatten=False):
    if args is None:
        mnist_train = False
    else:
        mnist_train = args.MnistTrain

    class MNIST(torchvision.datasets.MNIST):
        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            if flatten:
                img = img.view((-1))
            if res_selector is not None:
                return np.asarray([img, 0, target, index], dtype=object)[res_selector]
            return img, 0, target, index

    class Rev_MNIST(torchvision.datasets.MNIST):
        def __init__(self, *arg, **kwargs):
            warnings.warn('Rev_MNIST index')
            super(Rev_MNIST, self).__init__(*arg, **kwargs)

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            img, g, target, index = img, 1, target, index + (60000 if mnist_train else 10000)
            if flatten:
                img = img.view((-1))
            if res_selector is not None:
                return np.asarray([img, 1, target, index], dtype=object)[res_selector]
            return img, g, target, index

    class USPS(torchvision.datasets.USPS):
        def __init__(self, *arg, **kwargs):
            warnings.warn('USPS index')
            super(USPS, self).__init__(*arg, **kwargs)

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            if flatten:
                img = img.view((-1))
            img, g, target, index = img, 1, target, index + (60000 if mnist_train else 10000)
            if res_selector is not None:
                return np.asarray([img, 1, target, index], dtype=object)[res_selector]
            return img, g, target, index

    class MyImageFolder_0(torchvision.datasets.ImageFolder):
        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            img, g, target, index = sample, 0, target, index
            return img, g, target, index

    class MyImageFolder_1(torchvision.datasets.ImageFolder):
        def __init__(self, root, transform, ind_bias=2817, *args, **kwargs):
            super(MyImageFolder_1, self).__init__(root, transform, *args, **kwargs)
            self.ind_bias = ind_bias

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            img, g, target, index = sample, 1, target, index + self.ind_bias
            return img, g, target, index

    def featurelize(dataset):
        test_loader = data.DataLoader(dataset, batch_size=512, num_workers=NumWorkers)
        net = torchvision.models.resnet50(pretrained=True).cuda()
        # newnorm
        if args.dataset == 'MTFL':
            mean = 0.34949973225593567
            std = 0.3805956244468689
        elif args.dataset == 'Office':
            mean = 0.3970963954925537
            std = 0.43060600757598877
        else:
            raise NotImplementedError('args.dataset')

        # old norm
        # mean = 0.39618438482284546
        # std = 0.4320564270019531
        def forward(net, x):
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)
            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)
            x = net.avgpool(x)
            x = torch.flatten(x, 1)
            if args is None or args.FeatureType == 'Tanh':
                x = torch.nn.Tanh()(x)
            elif args.FeatureType == 'Gaussainlize':
                x = (x - mean) / std
            elif args.FeatureType == 'Normalize':
                x /= torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
            elif args.FeatureType == 'Default':
                pass
            elif args.FeatureType == 'GlS_GaussainlizeAndSigmoid':
                x = torch.nn.Sigmoid()((x - mean) / std)
            elif args.FeatureType == 'GlT_GaussainlizeAndTanh':
                x = torch.nn.Tanh()((x - mean) / std)
            elif args.FeatureType == 'Sigmoid':
                x = torch.nn.Sigmoid()(x)
            else:
                raise NotImplementedError('FeatureType')
            return x

        net.eval()
        feature_vec, type_vec, group_vec, idx_vec = [], [], [], []
        with torch.no_grad():
            for (x, g, y, idx) in test_loader:
                x = x.cuda()

                g = g.cuda()
                c = forward(net, x)

                feature_vec.extend(c.cpu().numpy())
                type_vec.extend(y.cpu().numpy())
                group_vec.extend(g.cpu().numpy())
                idx_vec.extend(idx.cpu().numpy())
        # feature_vec, type_vec, group_vec = np.array(feature_vec), np.array(
        #     type_vec), np.array(group_vec)
        x = torch.from_numpy(np.array(feature_vec))

        # x = (x - x.min()) / (x.max() - x.min())
        print('torch.mean(x)=={}, torch.std(x)=={}'.format(torch.mean(x), torch.std(x)))
        print('torch.min(x)=={}, torch.max(x)=={}'.format(torch.min(x), torch.max(x)))
        print('torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))=={}'.format(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))))
        # assert False
        img, g, target, index = x, torch.as_tensor(group_vec), torch.as_tensor(type_vec), torch.as_tensor(idx_vec)
        if flatten:
            img = img.view((len(img), -1))
        if res_selector is not None:
            item = np.asarray([img, g, target, index], dtype=object)[res_selector]
        else:
            item = [img, g, target, index]

        return data.TensorDataset(*item)

    if dataset == 'MNISTUSPS':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ])
        mnist = MNIST(train=True,
                      download=True,
                      root='../Datasets/Discrete/MNIST',
                      transform=transforms)
        usps = USPS(train=True,
                    download=True,
                    root='../Datasets/Discrete/USPS',
                    transform=transforms)
        if group == -1:
            data_set = data.ConcatDataset([mnist, usps])
        elif group == 0:
            data_set = mnist
        elif group == 1:
            data_set = usps
        else:
            raise NotImplementedError('group')

        class_num = 10
    elif dataset == 'ReverseMNIST':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5])
            ])

        class Reverse:
            def __call__(self, img):
                return 1 - img

        rev_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            Reverse(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        mnist = MNIST(train=mnist_train,
                      download=False,
                      root='../Datasets/Discrete/MNIST',
                      transform=transforms)
        rev_mnist = Rev_MNIST(train=mnist_train,
                              download=True,
                              root='../Datasets/Discrete/ReverseMNIST',
                              transform=rev_transforms)

        if group == -1:
            data_set = data.ConcatDataset([mnist, rev_mnist])
        elif group == 0:
            data_set = mnist
        elif group == 1:
            data_set = rev_mnist
        else:
            raise NotImplementedError('group')

        class_num = 10
    elif dataset == 'Office':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(224),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # newnorm
                torchvision.transforms.Normalize(mean=[0.7076, 0.7034, 0.7021],
                                                 std=[0.3249, 0.3261, 0.3275]),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                                  std=[0.229, 0.224, 0.225]),
            ])
        amazon = MyImageFolder_0('../Datasets/Discrete/Office/amazon',
                                 transform=transforms)
        webcam = MyImageFolder_1('../Datasets/Discrete/Office/webcam',
                                 transform=transforms)
        if group == -1:
            office = data.ConcatDataset([amazon, webcam])
        elif group == 0:
            office = amazon
        elif group == 1:
            office = MyImageFolder_1('../Datasets/Discrete/Office/webcam',
                                     transform=transforms, ind_bias=0)
        else:
            raise NotImplementedError('group')
        # print(office)
        # office
        # imgs = torch.stack([t[0] for t in office], dim=0)
        # print(imgs.shape)
        # t = torch.mean(imgs, dim=[0,2,3])
        # print(t)
        # t = torch.std(imgs, dim=[0,2,3])
        # print(t)
        data_set = featurelize(dataset=office)
        # assert False
        class_num = 31
    elif dataset == 'HAR':
        train_X = np.loadtxt('./Datasets/Discrete/HAR/train/X_train.txt')
        train_G = np.loadtxt('./Datasets/Discrete/HAR/train/subject_train.txt')
        train_Y = np.loadtxt('./Datasets/Discrete/HAR/train/y_train.txt')
        test_X = np.loadtxt('./Datasets/Discrete/HAR/test/X_test.txt')
        test_G = np.loadtxt('./Datasets/Discrete/HAR/test/subject_test.txt')
        test_Y = np.loadtxt('./Datasets/Discrete/HAR/test/y_test.txt')
        X = torch.cat((torch.from_numpy(train_X), torch.from_numpy(test_X)),
                      dim=0)
        G = torch.cat((torch.from_numpy(train_G), torch.from_numpy(test_G)),
                      dim=0) - 1
        Y = torch.cat((torch.from_numpy(train_Y), torch.from_numpy(test_Y)),
                      dim=0) - 1
        re_ind = np.argsort(G.numpy(), kind='stable') if group == -1 else G == group
        X = X[re_ind]
        G = G[re_ind]
        Y = Y[re_ind]

        #####
        # v = torch.sort(torch.sqrt(torch.sum(X ** 2, dim=0)))
        # print(v[0])
        # print(torch.amax(X , dim=0))
        # print(torch.amin(X , dim=0))
        # print(v[0][0])
        # print(v[0][0][0])
        # print(X.shape)
        # print(v[:50])
        # print(v[-50:])
        # assert False
        idx = torch.arange(0, X.shape[0])
        img, g, target, index = X.float(), G.long(), Y.long(), idx.long()
        # print("sensitive stats", torch.bincount(g))
        if res_selector is not None:
            item = np.asarray([img, g, target, index], dtype=object)[res_selector]
        else:
            item = [img, g, target, index]

        data_set = data.TensorDataset(*item)
        class_num = 6

    elif dataset == 'MTFL':
        if transforms is None:
            transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(224),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                                  std=[0.229, 0.224, 0.225]),
                # newnorm
                torchvision.transforms.Normalize(mean=[0.4951, 0.4064, 0.3579],
                                                 std=[0.2868, 0.2594, 0.2545]),
            ])
        use_1000 = True
        g0 = MyImageFolder_0(root='../Datasets/Discrete/MTFL/g0',
                             transform=transforms)
        g1 = MyImageFolder_1(ind_bias=1000 if use_1000 else 11346,
                             root='../Datasets/Discrete/MTFL/g1',
                             transform=transforms)
        if group == -1:
            mtfl = data.ConcatDataset([g0, g1])
        elif group == 0:
            mtfl = g0
        elif group == 1:
            mtfl = MyImageFolder_1(ind_bias=0,
                                   root='../Datasets/Discrete/MTFL/g1',
                                   transform=transforms)
        else:
            raise NotImplementedError('group')
        # imgs = torch.stack([t[0] for t in mtfl], dim=0)
        # print(imgs.shape)
        # t = torch.mean(imgs, dim=[0,2,3])
        # print(t)
        # t = torch.std(imgs, dim=[0,2,3])
        # print(t)
        # assert  False
        # office = amazon
        data_set = featurelize(dataset=mtfl)
        class_num = 2
    else:
        raise NotImplementedError('split')

    return data_set, class_num


def get_dataloader_discrete(dataset, dateset_mode=1, batch_size=512, **kwargs):
    data_set, class_num = get_dataset_discrete(dataset, **kwargs)

    ##normalize sensitive attr
    x_list = []
    g_list = []
    target_list = []
    index_list = []

    for x, g, target, index in data_set:
        x_list.append(x.numpy())
        if torch.is_tensor(g):
            g_list.append(g.numpy())
        else:
            g_list.append(g)
        if torch.is_tensor(target):
            target_list.append(target.numpy())
        else:
            target_list.append(target)
        if torch.is_tensor(index):
            index_list.append(index.numpy())
        else:
            index_list.append(index)


    data_size_all = len(index_list)

    x_np = np.array(x_list)
    g_np = np.array(g_list)
    g_n_np = (g_np - g_np.mean()) / g_np.std()
    target_np = np.array(target_list)
    index_list = np.array(index_list)

    item = [torch.from_numpy(x_np), torch.from_numpy(g_np), torch.from_numpy(g_n_np), torch.from_numpy(target_np),
                  torch.from_numpy(index_list)]
    data_set = data.TensorDataset(*item)


    dataset_all = (torch.from_numpy(x_np), torch.from_numpy(np.expand_dims(g_np, axis=1)),
                   torch.from_numpy(np.expand_dims(g_n_np, axis=1)))

    if dateset_mode == 1:
        train_loader = data.DataLoader(data_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True, num_workers=NumWorkers)

        test_loader = data.DataLoader(data_set, batch_size=batch_size*100, num_workers=NumWorkers)
        return train_loader, test_loader, class_num
    else:
        return dataset_all




def get_dataloader_continuous(dataset, dateset_mode=1, batch_size=512, **kwargs):
    if dataset == "Crime":
        data_set, class_num = genDataSetCrimeCommunity()
    elif dataset == "Obesity":
        data_set, class_num = genDataSetObesity()
    elif dataset == "Census":
        data_set, class_num = genDataSetCensus()

    elif dataset == "Patients":
        data_set, class_num = genDataSetPatient()
    elif dataset == "Adult":
        data_set, class_num =genDataSetAdult()


    ##normalize sensitive attr
    x_list = []
    g_list = []
    target_list = []
    index_list = []

    for x, g, target, index in data_set:
        x_list.append(x.numpy())
        if torch.is_tensor(g):
            g_list.append(g.numpy())
        else:
            g_list.append(g)
        if torch.is_tensor(target):
            target_list.append(target.numpy())
        else:
            target_list.append(target)
        if torch.is_tensor(index):
            index_list.append(index.numpy())
        else:
            index_list.append(index)

    x_np = np.array(x_list)
    g_np = np.array(g_list)
    g_n_np = g_np
    target_np = np.array(target_list, dtype=np.int32)
    index_list = np.array(index_list)

    item = [torch.from_numpy(x_np), torch.from_numpy(g_np), torch.from_numpy(g_n_np), torch.from_numpy(target_np),
                  torch.from_numpy(index_list)]
    data_set = data.TensorDataset(*item)


    dataset_all = (torch.from_numpy(x_np), torch.from_numpy(np.expand_dims(g_np, axis=1)),
                   torch.from_numpy(np.expand_dims(g_n_np, axis=1)))



    if dateset_mode == 1:
        train_loader = data.DataLoader(data_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True, num_workers=NumWorkers)

        test_loader = data.DataLoader(data_set, batch_size=batch_size * 100, num_workers=NumWorkers)
        return train_loader, test_loader, class_num
    else:
        return dataset_all


### discretize continuous variable data.
def get_dataloader_continuous_discretize(dataset, dateset_mode=1,num_intervals = 2, batch_size=512,  **kwargs):
    if dataset == "Crime":
        data_set, class_num = genDataSetCrimeCommunity()
    elif dataset == "Census":
        data_set, class_num = genDataSetCensus()


    ##normalize sensitive attr
    x_list = []
    g_list = []
    target_list = []
    index_list = []

    for x, g, target, index in data_set:
        x_list.append(x.numpy())
        if torch.is_tensor(g):
            g_list.append(g.numpy())
        else:
            g_list.append(g)
        if torch.is_tensor(target):
            target_list.append(target.numpy())
        else:
            target_list.append(target)
        if torch.is_tensor(index):
            index_list.append(index.numpy())
        else:
            index_list.append(index)

    x_np = np.array(x_list)
    g_np = np.array(g_list)
    g_n_np = np.array(g_list)
    target_np = np.array(target_list, dtype=np.int32)
    index_list = np.array(index_list)



    ##discretize attributes
    g_pd = pd.DataFrame(g_np)
    g_pd_d = pd.qcut(g_pd[0], num_intervals, labels=False, duplicates = 'drop')
    g_np = g_pd_d.to_numpy()

    item = [torch.from_numpy(x_np), torch.from_numpy(g_np), torch.from_numpy(g_n_np), torch.from_numpy(target_np),
                  torch.from_numpy(index_list)]
    data_set = data.TensorDataset(*item)


    dataset_all = (torch.from_numpy(x_np), torch.from_numpy(np.expand_dims(g_np, axis=1)),
                   torch.from_numpy(np.expand_dims(g_n_np, axis=1)))


    if dateset_mode == 1:
        train_loader = data.DataLoader(data_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True, num_workers=NumWorkers)

        test_loader = data.DataLoader(data_set, batch_size=batch_size * 100, num_workers=NumWorkers)
        return train_loader, test_loader, class_num
    else:
        return dataset_all



def get_dataloader_few_shot(dataset, data_size = 512, batch_size=64,  **kwargs):
    if dataset == "MNISTUSPS":
        data_set, class_num = get_dataset_discrete(dataset, **kwargs)
    elif dataset == "Census":
        data_set, class_num = genDataSetCensus()



    ##normalize sensitive attr
    x_list = []
    g_list = []
    target_list = []

    for x, g, target, _ in data_set:
        x_list.append(x.numpy())
        if torch.is_tensor(g):
            g_list.append(g.numpy())
        else:
            g_list.append(g)
        if torch.is_tensor(target):
            target_list.append(target.numpy())
        else:
            target_list.append(target)


    x_np = np.array(x_list)
    g_np = np.array(g_list)
    target_np = np.array(target_list)

    dataset_all = (torch.from_numpy(x_np), torch.from_numpy(g_np), torch.from_numpy(target_np))

    idx_shuffle = torch.randperm(len(dataset_all[1]))

    idx_train = idx_shuffle[0:data_size]

    idx_validate = idx_shuffle[data_size:data_size+1000]
    test_validate = len(idx_validate)

    idx_test = torch.randperm(len(dataset_all[1]))[data_size+1000:-1]
    test_size = len(idx_test)


    x_train = dataset_all[0][idx_train]
    g_np_train = dataset_all[1][idx_train]
    target_np_train = dataset_all[2][idx_train]

    x_validate = dataset_all[0][idx_validate]
    g_np_validate = dataset_all[1][idx_validate]
    target_np_validate = dataset_all[2][idx_validate]

    x_test = dataset_all[0][idx_test]
    g_np_test = dataset_all[1][idx_test]
    target_np_test = dataset_all[2][idx_test]




    item_train = [x_train, g_np_train, target_np_train]
    data_set_train = data.TensorDataset(*item_train)

    item_validate = [x_validate, g_np_validate, target_np_validate]
    data_set_validate = data.TensorDataset(*item_validate)

    item_test = [x_test, g_np_test, target_np_test]
    data_set_test = data.TensorDataset(*item_test)

    train_loader = data.DataLoader(data_set_train,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True, num_workers=NumWorkers)

    validate_loader = data.DataLoader(data_set_validate,
                                   batch_size=test_validate,
                                   shuffle=True,
                                   drop_last=True, num_workers=NumWorkers)



    test_loader = data.DataLoader(data_set_test, batch_size=test_size, num_workers=NumWorkers)

    return train_loader, validate_loader, test_loader, class_num



def get_dataloader_continuous2(dataset, dateset_mode=1, batch_size=512, **kwargs):

    if dataset == "Census":
        data_set, class_num = genDataSetCensus2()



    ##normalize sensitive attr
    x_list = []
    g_list = []
    target_list = []
    index_list = []
    g_ori_list = []

    for x, g, target, index, g_ori in data_set:
        x_list.append(x.numpy())
        if torch.is_tensor(g):
            g_list.append(g.numpy())
        else:
            g_list.append(g)
        if torch.is_tensor(target):
            target_list.append(target.numpy())
        else:
            target_list.append(target)
        if torch.is_tensor(index):
            index_list.append(index.numpy())
        else:
            index_list.append(index)

        if torch.is_tensor(g_ori):
            g_ori_list.append(g_ori.numpy())
        else:
            g_ori_list.append(g_ori)

    x_np = np.array(x_list)
    g_np = np.array(g_list)
    g_n_np = g_np
    target_np = np.array(target_list, dtype=np.int32)
    index_list = np.array(index_list)
    g_ori_list = np.array(g_ori_list)

    item = [torch.from_numpy(x_np), torch.from_numpy(g_np), torch.from_numpy(g_n_np), torch.from_numpy(target_np),
                  torch.from_numpy(index_list)]
    data_set = data.TensorDataset(*item)


    dataset_all = (torch.from_numpy(x_np), torch.from_numpy(np.expand_dims(g_np, axis=1)),
                   torch.from_numpy(np.expand_dims(g_n_np, axis=1)), g_ori_list)



    if dateset_mode == 1:
        train_loader = data.DataLoader(data_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True, num_workers=NumWorkers)

        test_loader = data.DataLoader(data_set, batch_size=batch_size * 100, num_workers=NumWorkers)
        return train_loader, test_loader, class_num
    else:
        return dataset_all



def genDataSetCrimeCommunity(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1):
    # create names
    names = []
    with open('./Datasets/Continuous/communities+and+crime/communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv('./Datasets/Continuous/communities+and+crime/communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(np.int)

    y = data[label].values
    for i in range(len(y)):
        # if y[i] <= 0.1:
        #     y[i] = 0
        # elif y[i] > 0.1 and y[i] <= 0.3:
        #     y[i] = 1
        # else:
        #     y[i] = 2
        if y[i] <= 0.15:
            y[i] = 0
        else:
            y[i] = 1


    to_drop += [label]

    s = data[sensitive_attribute].values
    # to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    index = np.arange(1, len(y) + 1, 1)

    dataset = data_utils.TensorDataset(torch.Tensor(x), torch.Tensor(s), torch.Tensor(y), torch.Tensor(index))
    classnum = 2
    return dataset, classnum


def genDataSetCensus():
    df = pd.read_csv('./Datasets/Continuous/Census/acs2015_census_tract_data.csv')
    # drop the NAs
    df = df.dropna()

    men_count = df['Men']
    women_count = df['Women']
    gender_ratio = men_count / (men_count + women_count)
    features = df.drop(['Income', 'State', 'CensusTract', 'County', 'Men', 'Women'], axis=1)
    sensitive_attr = gender_ratio
    n = len(features)
    features = np.atleast_2d(features).reshape(n, -1)
    sensitive_attr = np.atleast_2d(sensitive_attr).reshape(-1, 1)
    features = np.hstack((features, sensitive_attr))

    for n in range(features.shape[1]):
        features[:, n] = (features[:, n] - features[:, n].mean()) / features[:, n].std()
    sensitive_attr = sensitive_attr.squeeze()
    sensitive_attr = (sensitive_attr - sensitive_attr.mean()) / sensitive_attr.std()
    labels = np.atleast_2d(df['Income'].values).reshape(-1, 1).squeeze()
    labels[labels <= 50000] = 0
    labels[labels > 50000] = 1




    idx = torch.randperm(len(labels))[0:12500]
    features = features[idx]
    sensitive_attr = sensitive_attr[idx]
    labels = labels[idx]
    index = np.arange(1, len(labels) + 1, 1)




    # index = np.arange(1, len(labels) + 1, 1)
    dataset = data_utils.TensorDataset(torch.Tensor(features), torch.Tensor(sensitive_attr), torch.Tensor(labels),
                                       torch.Tensor(index))
    classnum = 2
    return dataset, classnum



def genDataSetCensus2():
    df = pd.read_csv('./Datasets/Continuous/Census/acs2015_census_tract_data.csv')
    # drop the NAs
    df = df.dropna()

    men_count = df['Men']
    women_count = df['Women']
    gender_ratio = men_count / (men_count + women_count)
    features = df.drop(['Income', 'State', 'CensusTract', 'County', 'Men', 'Women'], axis=1)
    sensitive_attr = gender_ratio
    n = len(features)
    features = np.atleast_2d(features).reshape(n, -1)
    sensitive_attr = np.atleast_2d(sensitive_attr).reshape(-1, 1)
    sensitive_attr_orig = sensitive_attr
    features = np.hstack((features, sensitive_attr))

    for n in range(features.shape[1]):
        features[:, n] = (features[:, n] - features[:, n].mean()) / features[:, n].std()
    sensitive_attr = sensitive_attr.squeeze()
    sensitive_attr = (sensitive_attr - sensitive_attr.mean()) / sensitive_attr.std()
    labels = np.atleast_2d(df['Income'].values).reshape(-1, 1).squeeze()
    labels[labels <= 50000] = 0
    labels[labels > 50000] = 1




    idx = torch.randperm(len(labels))[0:12500]
    features = features[idx]
    sensitive_attr = sensitive_attr[idx]
    labels = labels[idx]
    sensitive_attr_orig = sensitive_attr_orig[idx]
    index = np.arange(1, len(labels) + 1, 1)




    # index = np.arange(1, len(labels) + 1, 1)
    dataset = data_utils.TensorDataset(torch.Tensor(features), torch.Tensor(sensitive_attr), torch.Tensor(labels),
                                       torch.Tensor(index), torch.Tensor(sensitive_attr_orig))
    classnum = 2
    return dataset, classnum



def genDataSetAdult():
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.
    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''


    data = pd.read_csv(
        "./Datasets/Continuous/Adult/adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
    )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        "./Datasets/Continuous/Adult/adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        skiprows=1, header=None
    )
    data = pd.concat([data, data_test])


    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c


    name_sensitive_attr = 'Age'
    z = data[name_sensitive_attr].values
    features = data.drop([name_sensitive_attr], axis=1).values

    sensitive_attr = np.array(z).reshape(-1, 1)
    sensitive_attr = sensitive_attr.squeeze()
    sensitive_attr = (sensitive_attr - sensitive_attr.mean()) / sensitive_attr.std()


    # Care there is a final dot in the class only in test set which creates 4 different classes
    y = np.array([0.0 if (val == 0 or val == 1) else 1.0 for val in np.array(features)[:, -1]])

    for i in data.columns:
        data[i] = (data[i] - data[i].mean()) / data[i].std()
    features = np.array(data.values)
    features =features[:,0:-1]

    labels = np.array(y).reshape(-1, 1).squeeze()

    data_balance = True
    if data_balance:
        idx_0 = np.where(labels == 0)[0][0:6250]
        labels_0 = labels[idx_0]
        sensitive_attr_0 = sensitive_attr[idx_0]
        features_0 = features[idx_0]

        idx_1 = np.where(labels == 1)[0][0:6250]
        labels_1 = labels[idx_1]
        sensitive_attr_1 = sensitive_attr[idx_1]
        features_1 = features[idx_1]

    features = np.concatenate((features_0,features_1),axis = 0)
    labels = np.append(labels_0,labels_1)
    sensitive_attr = np.append(sensitive_attr_0, sensitive_attr_1)

    idx = torch.randperm(len(labels))
    features = features[idx]
    sensitive_attr = sensitive_attr[idx]
    labels = labels[idx]

    # print("labels: ", labels)
    # print("features: ", features)
    # print("sensitive_attr: ", sensitive_attr)






    index = np.arange(1, len(labels) + 1, 1)
    dataset = data_utils.TensorDataset(torch.Tensor(features), torch.Tensor(sensitive_attr), torch.Tensor(labels),
                                       torch.Tensor(index))

    classnum = 2


    return dataset, classnum






def genDataSetObesity():
    data = pd.read_csv('./Datasets/Continuous/ObesityDataSet/ObesityDataSet_raw_and_data_sinthetic.csv')
    s = data["Height"].to_numpy()
    s = (s - s.mean()) / s.std()

    label_list = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I",
                  "Obesity_Type_II", "Obesity_Type_III"]
    i = 0
    for label in label_list:
        data.replace(label, i, inplace=True)
        i = i + 1

    data.replace("yes", 1, inplace=True)
    data.replace("no", 0, inplace=True)
    data.replace("Sometimes", 1, inplace=True)
    data.replace("Frequently", 2, inplace=True)
    data.replace("Always", 3, inplace=True)

    data.replace("Walking", 0, inplace=True)
    data.replace("Bike", 1, inplace=True)
    data.replace("Public_Transportation", 2, inplace=True)
    data.replace("Motorbike", 3, inplace=True)
    data.replace("Automobile", 4, inplace=True)

    y = data["NObeyesdad"].to_numpy()

    fea = data.loc[:,
          ["family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE",
           "CALC", "Height"]]
    for n in fea.columns:
        fea[n] = (fea[n] - fea[n].mean()) / fea[n].std()

    x = np.array(fea.values)
    index = np.arange(1, len(y) + 1, 1)
    dataset = data_utils.TensorDataset(torch.Tensor(x), torch.Tensor(s), torch.Tensor(y), torch.Tensor(index))
    classnum = 7
    return dataset, classnum


def genDataSetPatient():
    data = pd.read_csv('./Datasets/Continuous/patients/processedData.csv')
    y = data["hospital_death"].to_numpy()
    s = data["age"].to_numpy()
    x = data.drop(["age", "hospital_death"], axis=1)
    x = np.array(x.values)

    index = np.arange(1, len(y) + 1, 1)

    dataset = data_utils.TensorDataset(torch.Tensor(x), torch.Tensor(s), torch.Tensor(y), torch.Tensor(index))
    classnum = 2

    return dataset, classnum



