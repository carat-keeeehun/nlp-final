import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset
from .vocab import Vocab
from datasets import load_dataset

DATA_PATH = './.datasets'

class dialogDataset(Dataset):
    """
    The text8 dataset consisting of 100M characters (with vocab size 27).
    We here split the dataset into (90M, 5M, 5M) characters for
    (train, val, test) as in [1,2,3].

    The sets are then split into chunks of equal length as specified by `seq_len`.
    The default is 256, corresponding to what was used in [1]. Other choices
    include 180, as [2] reports using.

    [1] Discrete Flows: Invertible Generative Models of Discrete Data
        Tran et al., 2019, https://arxiv.org/abs/1905.10347
    [2] Architectural Complexity Measures of Recurrent Neural Networks
        Zhang et al., 2016, https://arxiv.org/abs/1602.08210
    [3] Subword Language Modeling with Neural Networks
        Mikolov et al., 2013, http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf
    """

    def __init__(self, root=DATA_PATH, seq_len=256, split='train', download=False):
        assert split in {'train', 'validation', 'test'}
        self.root = os.path.join(root, 'dialog')
        dataset = load_dataset('daily_dialog', split=split)
        self.seq_len = seq_len
        self.split = split

        self.data = []
        stoi = {" ":0}
        c = 1

        for i in range(dataset.num_rows):
            for j in range(len(dataset[i]['dialog'])):
                if j!=len(dataset[i]['dialog'])-1:
                    m = dataset[i]['dialog'][j].lower()
                    n = dataset[i]['dialog'][j+1].lower()
                    if len(m)<=256 and len(n)<=256:
                        for s in m:
                            if not s in stoi:
                                stoi[s] = c
                                c += 1
                        for s in n:
                            if not s in stoi:
                                stoi[s] = c
                                c += 1
                        m = (m + " "*(256-len(m)))
                        n = (n + " "*(256-len(n)))
                        self.data.append((m, n))

        # Get vocabulary
        self.vocab = Vocab()
        vocab_file = os.path.join(self.root, 'vocab.json')
        if os.path.exists(vocab_file):
            self.vocab.load_json(self.root)
        else:
            self.vocab.fill(stoi)
            self.vocab.save_json(self.root)
        
        self._preprocessing()

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.seq_len

    def __len__(self):
        return len(self.data)

    def _preprocessing(self):
        temp = []
        for i, (c, x) in enumerate(self.data):
            c = torch.tensor([self.vocab.stoi[s] for s in c])
            x = torch.tensor([self.vocab.stoi[s] for s in x])
            temp.append((c,x))
        self.data = temp