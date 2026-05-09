import torch
import torch.nn as nn


class IndividualTF(nn.Module):
    """Per-region scoring variant of Raman-MRNet.

    Each of the three regions is encoded independently, and a small MLP
    (linear6 -> linear7 -> linear8) produces a scalar score per region. The
    scores are softmax-normalised across regions and used as attention weights
    for the weighted sum that feeds the final classifier.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(701, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 32)
        self.linear7 = nn.Linear(32, 16)
        self.linear8 = nn.Linear(16, 1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, flag=0):
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)

        out1 = out
        out1_1 = self.linear6(out1)
        out1_2 = self.linear7(out1_1)
        out1_3 = self.linear8(out1_2)

        out_weight = self.softmax(out1_3)
        out2 = out * out_weight

        out = torch.squeeze(torch.sum(out2, 2))
        out = self.classifier(out)
        out = torch.squeeze(out)
        return out, out_weight


class IndividualTF1(nn.Module):
    """Squeeze-and-Excitation variant of Raman-MRNet (paper version).

    The shared trunk encodes each region into a 64-d feature, then an SE module
    (3 -> 16 -> 3 with ReLU/Sigmoid) emits a per-region weight. The weighted
    region features are summed and passed through the classifier. The SE
    output (`excitation`) is also returned for visualisation.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(701, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)

        self.se_fc1 = nn.Linear(3, 16)
        self.se_fc2 = nn.Linear(16, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, input):
        """Forward pass.

        Args:
            input: tensor of shape (batch_size, 3, 701).

        Returns:
            logits: (batch_size, 2)
            excitation: (batch_size, 3) per-region attention weights.
        """
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)

        squeeze_out = torch.mean(out, dim=-1)

        excitation = self.relu(self.se_fc1(squeeze_out))
        excitation = self.sigmoid(self.se_fc2(excitation))

        out = out * excitation.unsqueeze(-1)
        out = torch.sum(out, dim=1)
        out = self.classifier(out)
        return out, excitation
