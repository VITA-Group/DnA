import sys
sys.path.append(".")
from torch import nn


class proj_head_simclr(nn.Module):
    def __init__(self, ch, output_cnt=None, finetuneMode=False):
        super(proj_head_simclr, self).__init__()
        self.in_features = ch
        self.finetuneMode = finetuneMode

        if output_cnt is None:
            output_cnt = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = nn.BatchNorm1d(ch)

        if not self.finetuneMode:
            self.fc2 = nn.Linear(ch, output_cnt, bias=False)
            self.bn2 = nn.BatchNorm1d(output_cnt)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # debug

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if not self.finetuneMode:
            x = self.fc2(x)
            x = self.bn2(x)

        return x
