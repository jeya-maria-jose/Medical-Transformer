from . import datasets


def build_dataloader(args, distributed=False):
    return datasets.__dict__[args.dataset](args, distributed)
