"""Oracle model definitions reused at inference time (no training).

Copied from the QRS artifact repository so this testing pipeline can load
the published CNN (binary) and GRU (multi-class) weights without depending
on the original training scripts.
"""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """Pure 1D CNN sequence classifier for binary MQTT violation detection."""

    def __init__(
        self,
        input_size,
        cnn_num_filters,
        cnn_kernel_size,
        cnn_pool_size,
        cnn_pooling_type="avg",
        num_classes=2,
        dropout=0.3,
    ):
        super(CNNClassifier, self).__init__()
        self.input_size = input_size
        self.cnn_num_filters = cnn_num_filters
        self.cnn_pooling_type = cnn_pooling_type

        cnn_layers = []
        in_channels = input_size
        for out_channels in cnn_num_filters:
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=cnn_kernel_size,
                    padding=cnn_kernel_size // 2,
                )
            )
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_num_filters[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        if self.cnn_pooling_type == "avg":
            pooled = cnn_out.mean(dim=2)
        else:
            pooled, _ = cnn_out.max(dim=2)
        out = self.dropout(pooled)
        return self.fc(out)


class GRUClassifier(nn.Module):
    """GRU sequence classifier for multi-class MQTT violation identification."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        num_classes,
        dropout=0.3,
    ):
        super(GRUClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_size, num_classes)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        if self.bidirectional:
            forward_hidden = h_n[-2, :, :]
            backward_hidden = h_n[-1, :, :]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = h_n[-1, :, :]
        out = self.dropout(final_hidden)
        return self.fc(out)
