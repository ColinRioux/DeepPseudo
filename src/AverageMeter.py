class AverageMeter:

    def __init__(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count