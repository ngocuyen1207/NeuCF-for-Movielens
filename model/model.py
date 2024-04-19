import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.retrieval import RetrievalNormalizedDCG

np.random.seed(123)
from torch import Tensor
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from data import *
from utils import *
class NCF(L.LightningModule):
    """ Neural Collaborative Filtering (NCF)    
    """
    
    def __init__(self):
        super().__init__()
        self.user_emb = nn.Embedding(7000, embedding_dim=32, max_norm=True)
        self.item_fc_1 = nn.Linear(in_features=54, out_features=32)
        # self.user_fc_2 = nn.Linear(in_features=128, out_features=64)
        # self.item_fc_2 = nn.Linear(in_features=128, out_features=64)
        self.bilinear = nn.Bilinear(in1_features=32, in2_features=32, out_features=1)
        # self.fc1 = nn.Linear(in_features=256, out_features=64)
        # self.fc2 = nn.Linear(in_features=64, out_features=16)
        # self.fc3 = nn.Linear(in_features=16, out_features=1)
        
    def forward(self, user_id, item_input):
        # user_id = user_id.squeeze()
        user_vector = nn.ReLU()(self.user_emb(user_id.squeeze()))
        item_vector = nn.ReLU()(self.item_fc_1(item_input))
        fusion_output = self.bilinear(user_vector.reshape(-1, 32), item_vector)
        pred = nn.Sigmoid()(fusion_output)
        # user_output = nn.ReLU()(self.user_fc_2(user_vector))
        # item_output = nn.ReLU()(self.item_fc_2(item_vector))
        # output = torch.cat((user_output, item_output, fusion_output), dim=1)
        # output = nn.ReLU()(self.fc1(output))
        # output = nn.ReLU()(self.fc2(output))
        # output = nn.ReLU()(self.fc3(output))
        # pred = nn.Sigmoid()(output)
        return pred
    
    def loss(self, preds: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        return nn.BCELoss()(preds, labels)
    
    def ndcg(self, predicted_labels, labels, user_ids):
        return RetrievalNormalizedDCG()(predicted_labels, labels, indexes=user_ids)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        user_input, item_input, user_ids, labels = batch
        predicted_labels = self(user_ids, item_input)
        # print('Percentiles ',[np.percentile(predicted_labels.numpy(), percentile) for percentile in [25,50,75,100]])
        loss = self.loss(predicted_labels, labels)
        ndcg = self.ndcg(predicted_labels, labels, user_ids)
        self.log_dict({'train_loss': loss,'train_ndcg': ndcg}, prog_bar=True, on_epoch=True, on_step=True,logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        user_input, item_input, user_ids, labels = batch
        predicted_labels = self(user_ids, item_input)
        loss = self.loss(predicted_labels, labels)
        ndcg = self.ndcg(predicted_labels, labels, user_ids)
        self.log_dict({'val_loss': loss,'val_ndcg': ndcg}, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return {"x": loss}

    def predict_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        user_input, item_input, user_ids, movie_ids = batch
        predicted_labels = self(user_ids, item_input)
        result = {'user_id': user_ids.squeeze().cpu().tolist(), 'movie_id': movie_ids.squeeze().cpu().tolist(), 'predictions': predicted_labels.squeeze().cpu().tolist()}
        upload_to_mongo(result)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(MovieLensDataset('train'), num_workers=4, batch_size=1024, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(MovieLensDataset('val'), num_workers=4, batch_size=1024, persistent_workers=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(MovieLensInferAllDataset(), num_workers=4, batch_size=1024, persistent_workers=True)
