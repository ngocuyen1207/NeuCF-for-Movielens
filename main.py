import logging
log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from data import MovieLensSingleUserDataset
from torch.utils.data import DataLoader
L.seed_everything(69)
import torch
from model import NCF
from fastapi import FastAPI, Depends
import os
import json
from utils import *
app = FastAPI()

PRETRAIN = r"model\checkpoint\neucfemb-epoch=16-val_loss=1.14-val_ndcg=0.78.ckpt"
PREDICTIONS = 'model/prediction.json'

import torch
from lightning.pytorch.callbacks import BasePredictionWriter

class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


def train():
    delete_all_predictions()
    model = NCF()
    checkpoint_callback = [
        ModelCheckpoint(
            dirpath="model//checkpoint",
            filename="neucfemb-{epoch:02d}-{val_loss:.2f}-{val_ndcg:.2f}",
        ),
        EarlyStopping(monitor="val_loss", mode="min"),
        # CustomWriter(output_dir="model/predictions/", write_interval="epoch")

    ]
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_progress_bar=True,
        callbacks=checkpoint_callback,
        deterministic=True,
        logger=pl_loggers.TensorBoardLogger(save_dir="model/"),
    )
    trainer.fit(model, ckpt_path=PRETRAIN)
    trainer.predict(model, return_predictions=False)

def train_one_user(user_id):
    model = NCF()
    dataloader = DataLoader(MovieLensSingleUserDataset(user_id, 'train'), num_workers=8)
    checkpoint_callback = [
        ModelCheckpoint(
            dirpath="model//checkpoint",
            filename="neucfemb-latest",
        ),
        EarlyStopping(monitor="val_loss", mode="min"),
    ]

    trainer = L.Trainer(
        accelerator="auto",
        enable_progress_bar=True,
        deterministic=True,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        callbacks=checkpoint_callback,
    )
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=PRETRAIN) 

@app.get("/{user_id}/{retrain}")
def infer_one_user(user_id: int = Depends(check_user_id_availability), retrain: bool=False):
    '''
    ## Infer cho một user \n
    **user_id**: User id \n
    **retrain**: Nếu user mới hay là cần train lại với rating mới thì đặt là True, sẽ mất khoảng 5 phút. Còn nếu không thì để False cho nhanh
    '''
    return {'user_id': [user_id,user_id, user_id],'movie_id': [1,2,3], 'predictions': [0.99, 0.97, 0.96]}
    model = NCF()
    dataloader = DataLoader(MovieLensSingleUserDataset(user_id, 'infer'), num_workers=8, persistent_workers=True)

    if retrain:
        train_one_user(user_id)
        trainer = L.Trainer(
            accelerator="auto",
            enable_progress_bar=True,
            deterministic=True,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            inference_mode=True,
        )
        predictions = trainer.predict(model=model, dataloaders=dataloader, return_predictions=True,ckpt_path=PRETRAIN)
        return predictions
    else:
        with open(PREDICTIONS, 'r') as json_file:
            predictions = pd.DataFrame(json.load(json_file))
        predictions = predictions[predictions.user_id == user_id]
        predictions = {'user_id': predictions.user_id.tolist(), 'movie_id': predictions.movie_id.tolist(), 'predictions': predictions.predictions.tolist()}   
        return predictions
            

if __name__ == "__main__":
    train()
    # infer_one_user(2, False)
