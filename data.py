# data.py
import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
opt = {
    'data_folder': '/content/data/images/CloneImages',
    'num_classes': 2,
    'batch_size': 128,
    'val_batch_size': 512,
}


def getTrainValLoader(opt):
  image_transforms = transforms.Compose([
      transforms.RandomRotation(10),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  target_transforms = transforms.Lambda(lambda y: torch.zeros(opt['num_classes'],
                                                              dtype=torch.float).scatter_(0,
                                                                                         torch.tensor(y),
                                                                                          value=1))
  dataset = ImageFolder(opt['data_folder'],
                        transform=image_transforms,
                        target_transform=target_transforms
                        )

  validation_split = int(len(dataset) * opt['validation_split'])
  train_split = len(dataset) - validation_split
  train_set, valid_set = random_split(dataset, [train_split, validation_split])

  train_loader = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_set, batch_size=opt['val_batch_size'], shuffle=False)
  return train_loader, valid_loader
#getTrainValLoader(opt)
