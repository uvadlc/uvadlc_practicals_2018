import os
import errno

import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
from PIL import Image


BMNIST_BASE_URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/" \
                  "binarized_mnist/binarized_mnist_{}.amat"


class BMNIST(data.Dataset):
    """ BINARY MNIST """

    urls = [BMNIST_BASE_URL.format(split) for split in
            ['train', 'valid', 'test']]
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "train.pt"
    val_file = "val.pt"
    test_file = "test.pt"

    def __init__(self, root, split='train', transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        if split not in ('train', 'val', 'test'):
            raise ValueError('split should be one of {train, val, test}')

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        data_file = {'train': self.training_file,
                     'val': self.val_file,
                     'test': self.test_file}[split]
        path = os.path.join(self.root, self.processed_folder, data_file)
        self.data = torch.load(path)

    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.float().numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        processed_folder = os.path.join(self.root, self.processed_folder)
        train_path = os.path.join(processed_folder, self.training_file)
        val_path = os.path.join(processed_folder, self.val_file)
        test_path = os.path.join(processed_folder, self.test_file)
        return os.path.exists(train_path) and os.path.exists(val_path) and \
            os.path.exists(test_path)

    def _read_raw_image_file(self, path):

        raw_file = os.path.join(self.root, self.raw_folder, path)
        all_images = []
        with open(raw_file) as f:
            for line in f:
                im = [int(x) for x in line.strip().split()]
                assert len(im) == 28 ** 2
                all_images.append(im)
        return torch.from_numpy(np.array(all_images)).view(-1, 28, 28)

    def download(self):
        """
        Download the BMNIST data if it doesn't exist in
        processed_folder already.
        """

        if self._check_exists():
            return

        # Create folders
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, root=os.path.join(self.root, self.raw_folder),
                         filename=filename, md5=None)

        # process and save as torch files
        print('Processing raw data..')

        training_set = self._read_raw_image_file('binarized_mnist_train.amat')
        val_set = self._read_raw_image_file('binarized_mnist_valid.amat')
        test_set = self._read_raw_image_file('binarized_mnist_test.amat')

        processed_dir = os.path.join(self.root, self.processed_folder)
        with open(os.path.join(processed_dir, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(processed_dir, self.val_file), 'wb') as f:
            torch.save(val_set, f)
        with open(os.path.join(processed_dir, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Completed data download.')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        tmp_ = self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        fmt_str += '{0}{1}\n'.format(tmp, tmp_)
        return fmt_str


def bmnist(root='./data/', batch_size=128, download=True):

    data_transforms = transforms.Compose([transforms.ToTensor()])

    train_set = BMNIST(root, 'train', data_transforms, download)
    val_set = BMNIST(root, 'val', data_transforms, download)
    test_set = BMNIST(root, 'test', data_transforms, download)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=10)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=10)

    return trainloader, valloader, testloader
