import torch
from torch import nn
from torchmetrics import Accuracy
from typing import Optional, Tuple

class TinyImageNetModel(nn.Module):
    # define optionals and return types
    def __init__(self, lr: Optional[float], device) -> None:
        super().__init__()

        self.lr: float = lr or 0.0001
        self.device = device

        # model layers
        self.tinyimgnet_model = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (5, 5)),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),            
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Flatten(),

            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 200),
            # I guess you can't softmax because CrossEntropyLoss expects a raw data and not a classification on a scale of 0 to 1 (which is what adding the softmax produces)
            # nn.Softmax(dim=1)
        )

    def forward(self, input) -> torch.Tensor:
        output = self.tinyimgnet_model(input)
        return output
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self._step("train", batch)
    
    def _step(self, step_name, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        data, labels = batch
        # data = data.to(self.device)
        # labels = labels.to(self.device)

        f_loss = torch.nn.CrossEntropyLoss()

        pred = self.forward(data).to(self.device)
        loss = f_loss(pred, labels).to(self.device)

        acc = Accuracy(task="multiclass", num_classes=200).to(self.device)
        acc_val = acc(pred.argmax(dim=1), labels)
        # print(f'{step_name}_loss = {loss}\t // {step_name}_acc = {acc_val}')

        return (loss, acc_val)
    
    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters(), lr=self.lr)
    
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
