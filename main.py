import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from model import NCF


def train():
    model = NCF()
    checkpoint_callback = [
        ModelCheckpoint(
            dirpath="model//checkpoint",
            filename="neucfbig-{epoch:02d}-{val_loss:.2f}-{val_ndcg:.2f}",
        ),
        EarlyStopping(monitor="val_loss", mode="min"),
    ]
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        enable_progress_bar=True,
        callbacks=checkpoint_callback,
        logger=pl_loggers.TensorBoardLogger(save_dir="model/"),
    )
    trainer.fit(model, ckpt_path="last")


if __name__ == "__main__":
    train()
