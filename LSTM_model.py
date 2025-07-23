# Code for the LSTM model

from multiprocessing import pool
import torch
import torch.nn as nn
from GAT_model import GATModel

class LSTMModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, lsmt_hidden, num_classes):
        super(LSTMModel, self).__init__()
        self.gat_encoder = GATModel(in_channels, hidden_channels)
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=lsmt_hidden, batch_first=True, bidirectional=True)
        self.dim_red = nn.Sequential(
            nn.Linear(int(lsmt_hidden*2), lsmt_hidden),
            nn.ReLU(),
            nn.Linear(lsmt_hidden, int(lsmt_hidden/3)),
            nn.ReLU,
        )
        self.classifier = nn.Linear(lsmt_hidden/3, num_classes)

    def forward(self, sequences):

        # sequences is a list of length BATCH_SIZE
        # each item is a list of slice graphs for a patient scan
        batch_embeddings = []

        for sequence in sequences:  # loop over each patient in batch
            slice_embeddings = []

            for graph in sequence:  # loop over slices in that patient's scan
                graph = graph.to(next(self.parameters()).device)
                
                graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(graph.x.device)

                emb = self.gat_encoder(graph.x, graph.edge_index, graph.batch)  # [1, gat_hidden]
                slice_embeddings.append(emb)

            slice_embeddings = torch.cat(slice_embeddings, dim=0)  # [num_slices, gat_hidden]
            batch_embeddings.append(slice_embeddings.unsqueeze(0))  # [1, num_slices, gat_hidden]

        # [batch_size, num_slices, gat_hidden]
        batch_embeddings = torch.cat(batch_embeddings, dim=0)

        # # LSTM output: take final hidden state
        # _, (h_n, _) = self.lstm(batch_embeddings)  # h_n: [1, batch_size, lstm_hidden]
        # output = self.classifier(h_n.squeeze(0))   # [batch_size, num_classes]

        lstm_out, _ = self.lstm(batch_embeddings)
        pooled = torch.mean(lstm_out, dim=1)
        reduced = self.dim_red(pooled)
        output = self.classifier(reduced)

        return output