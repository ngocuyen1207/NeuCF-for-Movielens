import lightning as L
L.seed_everything(69)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers
from model import NCF
from data import MovieLensSingleUserDataset
import asyncio
from torch.utils.data import DataLoader

from fastapi import FastAPI, Depends
from utils import check_user_id_availability,  delete_all_predictions, get_predictions_from_mongo
PRETRAIN = r"model\checkpoint\neucfemb-epoch=13-val_loss=0.62-val_ndcg=0.79.ckpt"

app = FastAPI()

@app.post("/train")
async def train_all_users():
    delete_all_predictions()
    model = NCF()
    checkpoint_callback = [
        ModelCheckpoint(
            save_top_k=-1,
            monitor='val_loss',
            dirpath="model//checkpoint",
            filename="neucfemb-{epoch:02d}-{val_loss:.2f}-{val_ndcg:.2f}",
        ),
        EarlyStopping(monitor="val_loss", mode="min"),
    ]
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_progress_bar=True,
        callbacks=checkpoint_callback,
        deterministic=True,
        logger=pl_loggers.TensorBoardLogger(save_dir="model/"),
    )
    trainer.fit(model, 
                ckpt_path=PRETRAIN
                )
    trainer.predict(model, return_predictions=False, ckpt_path=PRETRAIN)

def train_one_user(user_id):
    model = NCF()
    train_dataloader = DataLoader(MovieLensSingleUserDataset(user_id, 'train'), num_workers=8)
    checkpoint_callback = [
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
    trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=PRETRAIN) 
    infer_dataloader = DataLoader(MovieLensSingleUserDataset(user_id, 'infer'), num_workers=8, batch_size=2048)
    trainer.predict(model, dataloaders=infer_dataloader, return_predictions=False)

@app.get("/{user_id}/{retrain}")
async def infer_one_user(user_id: int, retrain: bool):
    '''
    ## Infer cho một user \n
    **user_id**: User id \n
    **retrain**: Nếu user mới hay là cần train lại với rating mới thì đặt là True, sẽ mất khoảng 5 phút. Còn nếu không thì để False cho nhanh
    '''

    if retrain:
        train_one_user(user_id)
    check_user_id_availability(user_id)
    return get_predictions_from_mongo(user_id)
            
if __name__=='__main__':
    train_one_user(100)