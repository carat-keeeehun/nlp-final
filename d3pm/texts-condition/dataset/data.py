import torch
from torch.utils.data import DataLoader, ConcatDataset
from .dataset_text8 import Text8Dataset
from .dataset_enwik8 import EnWik8Dataset
from .dataset_dialog import dialogDataset

dataset_choices = {'text8_256', 'enwik8_blocksparse', 'dialog'}

def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='dialog', choices=dataset_choices)
    parser.add_argument('--validation', type=eval, default=True)

    # Train params
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=True)


def get_data_id(args):
    return args.dataset


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    if args.dataset == 'text8_256':
        train = Text8Dataset(seq_len=256, split='train', download=True)
        valid = Text8Dataset(seq_len=256, split='valid')
        test = Text8Dataset(seq_len=256, split='test')
        data_shape = (256,)
        num_classes = 27
    elif args.dataset == 'enwik8_blocksparse':
        train = EnWik8Dataset(seq_len=320, split='train', download=True)
        valid = EnWik8Dataset(seq_len=320, split='valid')
        test = EnWik8Dataset(seq_len=320, split='test')
        data_shape = (320,)
        num_classes = 256
    elif args.dataset == 'dialog':
        train = dialogDataset(seq_len=256, split='train')
        valid = dialogDataset(seq_len=256, split='validation')
        test = dialogDataset(seq_len=256, split='test')
        data_shape = (256,)
        num_classes = 73

    # Data Loader
    if args.validation:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        eval_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        dataset_train = ConcatDataset([train, valid])
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        eval_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_loader, eval_loader, data_shape, num_classes
