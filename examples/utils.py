class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class RunningMaximumMeter(object):
    """Computes and stores the maximum value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.max = float('-inf')

    def update(self, val):
        if self.val is None:
            self.max = val
        else:
            self.max = max(self.max, val)
        self.val = val

class AverageMeter(object):
    """Computes and stores the cumulative average, sum, count, and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
