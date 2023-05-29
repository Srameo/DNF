import torch.nn as nn
from torch.nn import MSELoss, L1Loss

class Losses(nn.Module):
    def __init__(self, classes, names, weights, positions, gt_positions):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.names = names
        self.weights = weights
        self.positions = positions
        self.gt_positions = gt_positions
        for class_name in classes:
            module_class = eval(class_name)
            self.module_list.append(module_class())

    def __len__(self):
        return len(self.names)

    def forward(self, outputs, targets):
        losses = []
        for i in range(len(self.names)):
            loss = self.module_list[i](outputs[self.positions[i]], targets[self.gt_positions[i]]) * self.weights[i]
            losses.append(loss)
        return losses

def build_loss(config):
    loss_names = config['types']
    loss_classes = config['classes']
    loss_weights = config['weights']
    loss_positions = config['which_stage']
    loss_gt_positions = config['which_gt']
    assert len(loss_names) == len(loss_weights) == \
           len(loss_classes) == len(loss_positions) == \
           len(loss_gt_positions)
    criterion = Losses(classes=loss_classes, names=loss_names,
                          weights=loss_weights, positions=loss_positions,
                          gt_positions=loss_gt_positions)
    return criterion
