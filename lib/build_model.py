from . import models


def build_model(args):
    model = models.__dict__[args.model](num_classes=args.num_classes)
    return model
