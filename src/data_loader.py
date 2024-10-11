import torchaudio
from torch.utils.data import DataLoader

def load_librispeech(batch_size=8, split='train-clean-100', root='./data'):
    dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
