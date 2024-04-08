import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from build_vocab import Vocabulary

class CaptionDataset(Dataset):
    def __init__(self, image_dir, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __getitem__(self, index):
        image_name = f"{index+1}.{'png' if index == 0 else 'jpg'}"
        image_path = os.path.join(self.image_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to tensor of word ids.
        caption = f"<start> This is a sample caption. <end>"
        caption = caption.split()
        caption = [self.vocab.word2idx[word] for word in caption]
        target = torch.Tensor(caption)

        return image, target, len(caption)

    def __len__(self):
        return len(self.image_files)


def get_loader(image_dir, vocab, transform, batch_size=32, shuffle=True, num_workers=2):
    dataset = CaptionDataset(image_dir, vocab, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
