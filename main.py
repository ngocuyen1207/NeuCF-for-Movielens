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
from fastapi import FastAPI

app = FastAPI()

MODEL_TO_LOAD = r"model\checkpoint\neucfemb-epoch=07-val_loss=0.38-val_ndcg=0.76.ckpt"


def train():
    model = NCF()
    checkpoint_callback = [
        ModelCheckpoint(
            dirpath="model//checkpoint",
            filename="neucfemb-{epoch:02d}-{val_loss:.2f}-{val_ndcg:.2f}",
        ),
        EarlyStopping(monitor="val_loss", mode="min"),
    ]
    trainer = L.Trainer(
        max_epochs=50,
        overfit_batches=0.07,
        accelerator="auto",
        enable_progress_bar=True,
        callbacks=checkpoint_callback,
        deterministic=True,
        logger=pl_loggers.TensorBoardLogger(save_dir="model/"),
    )
    trainer.fit(model, ckpt_path="last")

def train_one_user(user_id):
    model = NCF()
    dataloader = DataLoader(MovieLensSingleUserDataset(user_id, 'train'), num_workers=8)
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        enable_progress_bar=True,
        deterministic=True,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=pl_loggers.TensorBoardLogger(save_dir="model/"),
    )
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=MODEL_TO_LOAD)
    return trainer

@app.get("/{user_id}/{retrain}")
def infer_one_user(user_id, retrain=True):
    model = NCF()
    dataloader = DataLoader(MovieLensSingleUserDataset(user_id, 'infer'), num_workers=8, persistent_workers=True)
    if retrain:
        trainer = train_one_user(user_id)
        y_hat = trainer.predict(dataloaders=dataloader, return_predictions=True,ckpt_path=MODEL_TO_LOAD)
    else:
        model = NCF.load_from_checkpoint(MODEL_TO_LOAD)
        model.eval()
        with torch.no_grad():
            y_hat = model(dataloader)  
    print(y_hat)
    return y_hat

if __name__ == "__main__":
    # train()
    infer_one_user(2, True)
