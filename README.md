# Extentions

Some extentions for pytorch and pytorch-ligthning code, like callacks, extra datasets.

# Usage
```
from extentions.callbacks import DiffEarlyStopping, EarlyStopping

callbacks = [
    DiffEarlyStopping(
        monitor1="val_loss",
        monitor2="train_loss",
        diff_threshold=0.05, # like val_loss=0.09, train_loss=0.04
        patience=5,
        verbose=True
    ),
    EarlyStopping(
        monitor="val_acc",
        min_delta=0.0,
        mode='max',
        stopping_threshold=99.99,
        patience=5,
        verbose=True
    )
]
```
