import torch.optim as optim


def build_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    else:
        raise AssertionError
    return optimizer

