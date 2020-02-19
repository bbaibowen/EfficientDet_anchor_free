import os

import torch
from torchvision import datasets

from tool.boxlist import BoxList


CLASS_NAME = [
    '__background__',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]


def has_only_empty_bbox(annot):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


def has_valid_annotation(annot):
    if len(annot) == 0:
        return False

    if has_only_empty_bbox(annot):
        return False

    return True


class COCODataset(datasets.CocoDetection):
    def __init__(self, path, split, transform=None):
        root = os.path.join(path, '{}2017'.format(split))
        annot = os.path.join(path, 'annotations', 'instances_{}2017.json'.format(split))

        super().__init__(root, annot)

        self.ids = sorted(self.ids)

        if split == 'train':
            ids = []

            for id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=id, iscrowd=None)
                annot = self.coco.loadAnns(ann_ids)

                if has_valid_annotation(annot):
                    ids.append(id)

            self.ids = ids

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.id2img = {k: v for k, v in enumerate(self.ids)}

        self.transform = transform

    def __getitem__(self, index):
        img, annot = super().__getitem__(index)

        annot = [o for o in annot if o['iscrowd'] == 0]

        boxes = [o['bbox'] for o in annot]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [o['category_id'] for o in annot]
        classes = [self.category2id[c] for c in classes]
        classes = torch.tensor(classes)
        target.fields['labels'] = classes

        target.clip(remove_empty=True)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, index

    def get_image_meta(self, index):
        id = self.id2img[index]
        img_data = self.coco.imgs[id]

        return img_data


class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = (max_size[1] | (stride - 1)) + 1
        max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)


def collate_fn(config):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], config.size_divisible)
        targets = batch[1]
        ids = batch[2]

        return imgs, targets, ids

    return collate_data
