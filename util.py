'''Modified from https://github.com/alinlab/LfF/blob/master/util.py'''

import torch

class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9, device=None):
        self.label = label.cpu()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cpu()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        data = data.cpu()
        self.parameter = self.parameter.cpu()
        self.updated = self.updated.cpu()
        index = index.cpu()

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data

        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()


class Hook:
    def __init__(self, module, backward=False):
        self.feature = []
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.feature.append(output)

    def close(self):
        self.hook.remove()