import torch
import torchvision
from torchvision import datasets, transforms


def imagenet1k(args, distributed=False):
    train_dirs = args.train_dirs
    val_dirs = args.val_dirs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    color_jitter = args.color_jitter

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    process = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter:
        process += [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
    process += [
        transforms.ToTensor(),
        normalize
    ]

    transform_train = transforms.Compose(process)

    train_set = datasets.ImageFolder(train_dirs,
                                     transform=transform_train)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                                               sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    transform_val = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize])

    val_set = datasets.ImageFolder(root=val_dirs,
                                   transform=transform_val)

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
                                             sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    return train_loader, train_sampler, val_loader, val_sampler
