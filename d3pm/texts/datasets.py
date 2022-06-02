import json
import numpy as np
import os
import torch
import torch.nn as nn
import urllib.request
import zipfile

from torch.utils.data import Dataset


DATA_PATH = './datasets'


class Vocab():
    def __init__(self, stoi={}):
        self.fill(stoi)

    def fill(self, stoi):
        self.stoi = stoi
        self.itos = {i:s for s,i in stoi.items()}

    def save_json(self, path):
        if not os.path.exists(path): os.makedirs(path)
        vocab_file = os.path.join(path, 'vocab.json')
        with open(vocab_file, 'w') as f:
            json.dump(self.stoi, f, indent=4)

    def load_json(self, path):
        vocab_file = os.path.join(path, 'vocab.json')
        with open(vocab_file, 'r') as f:
            stoi = json.load(f)
        self.fill(stoi)

    def string_to_idx(self, string):
        assert isinstance(string, str)
        return [self.stoi[s] for s in string]

    def idx_to_string(self, idx):
        assert isinstance(idx, list)
        count_err = np.sum([1 for i in idx if i not in self.itos])
        if count_err > 0:
            print(f'Warning, {count_err} decodings were not in vocab.')
            print(set([i for i in idx if i not in self.itos]))
        return ''.join([self.itos[i] if i in self.itos else '?' for i in idx])

    def encode(self, text, padding_value=0):
        assert isinstance(text, list)
        length = torch.tensor([len(string) for string in text])
        tensor_list = [torch.tensor(self.string_to_idx(string)) for string in text]
        tensor = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)
        return tensor, length

    def decode(self, tensor, length):
        assert torch.is_tensor(tensor)
        assert tensor.dim() == 2, 'Tensor should have shape (batch_size, seq_len)'
        text = [self.idx_to_string(tensor[b][:length[b]].tolist()) for b in range(tensor.shape[0])]
        return text


class Text8(Dataset):
    """
    The text8 dataset consisting of 100M characters (with vocab size 27).
    We here split the dataset into (90M, 5M, 5M) characters for
    (train, val, test) as in [1,2,3].

    The sets are then split into chunks of equal length as specified by `seq_len`.
    The default is 256, corresponding to what was used in [1]. Other choices
    include 180, as [2] reports using.
    """

    def __init__(self, root=DATA_PATH, seq_len=256, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(root, 'text8')
        self.seq_len = seq_len
        self.split = split

        if not os.path.exists(self.raw_file):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it.')

        # Get vocabulary
        self.vocab = Vocab()
        vocab_file = os.path.join(self.root, 'vocab.json')
        if os.path.exists(vocab_file):
            self.vocab.load_json(self.root)
        else:
            stoi = self._create_stoi()
            self.vocab.fill(stoi)
            self.vocab.save_json(self.root)

        # Preprocess data
        if not os.path.exists(self.processed_file(split)):
            self._preprocess_data(split)

        # Load data
        self.data = torch.load(self.processed_file(split))

    def __getitem__(self, index):
        return self.data[index], self.seq_len

    def __len__(self):
        return len(self.data)

    def _create_stoi(self):
        rawdata = zipfile.ZipFile(self.raw_file).read('text8').decode('utf-8')
        s = sorted(list(set(rawdata)))
        stoi = {s[i]: i for i in range(len(s))}
        return stoi

    def _preprocess_data(self, split):
        # Read raw data
        rawdata = zipfile.ZipFile(self.raw_file).read('text8').decode('utf-8')

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:90000000]
        elif split == 'valid':
            rawdata = rawdata[90000000:95000000]
        elif split == 'test':
            rawdata = rawdata[95000000:]

        # Encode characters
        data = torch.tensor([self.vocab.stoi[s] for s in rawdata])

        # Split into chunks
        data = data[:self.seq_len*(len(data)//self.seq_len)]
        data = data.reshape(-1, self.seq_len)

        # Save processed data
        torch.save(data, self.processed_file(split))

    @property
    def raw_file(self):
        return os.path.join(self.root, 'text8.zip')

    def processed_file(self, split):
        return os.path.join(self.root, 'processed_{}.pt'.format(split))

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading text8...')
        url = 'http://mattmahoney.net/dc/text8.zip'
        print('Downloading from {}...'.format(url))
        urllib.request.urlretrieve(url, self.raw_file)
        print('Saved to {}'.format(self.raw_file))


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader, ConcatDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='text8')
    parser.add_argument('--validation', type=eval, default=True)
    # Train params
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)

    args = parser.parse_args()

    if args.dataset == "text8":
        train = Text8(seq_len=256, split="train", download=True)
        valid = Text8(seq_len=256, split="valid")
        test = Text8(seq_len=256, split="test")
        data_shape = (256,)
        num_classes = 27

    # Data Loader
    if args.validation:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        eval_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        dataset_train = ConcatDataset([train, valid])
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        eval_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    for i, (x, length) in enumerate(train_loader):
        print(x)
        print(length)
        break
