import torch


class ReactiveBaseline():
    def __init__(self, option, update_rate):
        self.option = option
        self.update_rate = update_rate
        self.value = torch.zeros(1)
        self.value = self.value.to(option.device)
        self.saved = self.value

    def get_baseline_value(self):
        return self.value
        # original 0

    def update(self, target):
        self.saved = self.value
        self.value = torch.add((1 - self.update_rate) * self.value, self.update_rate * target)

    def reset(self):
        self.value = self.saved
