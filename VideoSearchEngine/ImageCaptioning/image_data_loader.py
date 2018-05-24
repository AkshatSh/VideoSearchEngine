import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import json


class COCODataset(data.dataset):
    def __init__(self, root, coco_annotation, vocab, detection_result, transform=None):
        self.root = root
        self.coco = COCO(coco_annotation)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab 
        self.transform = transform 
        with open(detection_result, 'r') as f:
            self.detection_result = json.load(f)
        self.locations = {result['id'] : result['bboxes'] for result in self.detection_result}
        self.labels = {result['id'] : result['full_categories'] for result in self.detection_result}
    
    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        # apply the transform if it exists
        if self.transform is not None:
            image = self.transform(image)
        
        tokens = nltk.tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        labels = self.labels[img_id]
        locations = self.locations[img_id]
        if len(labels) != len(locations):
            raise ValueError("number of labels nust be equal to number of locations")
        if len(labels) == 0:
            labels = [0]
            locations = [0]
        return image, target, labels, locations
    
    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
        # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, label_seqs, location_seqs = zip(*data)
    assert len(label_seqs) > 0
    assert len(label_seqs) == len(location_seqs)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    label_seq_lengths = [len(label_seq) for label_seq in label_seqs]
    label_seq_data = torch.zeros(len(label_seqs), max(label_seq_lengths)).long()
    for i, label_seq in enumerate(label_seqs):
        label_seq_data[i, :len(label_seq)] = torch.LongTensor(label_seq[:len(label_seq)])

    location_seq_data = torch.zeros(len(location_seqs), max(label_seq_lengths), 4)
    for i, location_seq in enumerate(location_seqs):
        for j in range(len(location_seq)):
            coords = decode_location(location_seq[j])
            location_seq_data[i, j] = coords

    return images, targets, lengths, label_seq_data, location_seq_data, label_seq_lengths

def decode_location(location):
    x = location // 1e9
    y = (location % 1e9) // 1e6
    width = (location % 1e6) // 1e3
    height = location % 1e3
    return torch.Tensor((x / 608, y / 608, width / 608, height / 608))

def get_loader(root, coco_annotation, vocab, coco_detection_result, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       coco_annotation=coco_annotation,
                       vocab=vocab,
                       coco_detection_result=coco_detection_result,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader