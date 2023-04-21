import os
import pandas as pd
import torchvision.transforms as T
import torch.utils.data as td
from multilabel.loader import MultiLabelDataset


def load_data(path, batch_size, input_size, norm_arr, num_workers=0):

    transform_dict = {
        "train": T.Compose(
            [
                T.Resize(size=(input_size, input_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(30),
                T.RandomAdjustSharpness(sharpness_factor=2),
                T.RandomAutocontrast(p=0.5),
                T.ToTensor(),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.Normalize(*norm_arr),
            ]
        ),
        "test_val": T.Compose(
            [
                T.Resize(size=(input_size, input_size)),
                T.ToTensor(),
                T.Normalize(*norm_arr),
            ]
        ),
    }

    # Read metadata, replace nan with 0 and then convert all non 0 values to 1.
    train = pd.read_csv(os.path.join(path, "train.csv")).set_index("Path").fillna(0).astype(bool).astype(int)
    test = pd.read_csv(os.path.join(path, "test.csv")).set_index("Path").fillna(0).astype(bool).astype(int)
    val = pd.read_csv(os.path.join(path, "val.csv")).set_index("Path").fillna(0).astype(bool).astype(int)

    train_dataset = MultiLabelDataset(
        root=os.path.join(path, "pre_proc_images", "train"),
        dataframe=train,
        transform=transform_dict["train"],
    )
    val_dataset = MultiLabelDataset(
        root=os.path.join(path, "pre_proc_images", "val"),
        dataframe=val,
        transform=transform_dict["test_val"],
    )
    test_dataset = MultiLabelDataset(
        root=os.path.join(path, "pre_proc_images", "test"),
        dataframe=test,
        transform=transform_dict["test_val"],
    )

    data_loader_train = td.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    data_loader_val = td.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    data_loader_test = td.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": data_loader_train,
        "val": data_loader_val,
        "test": data_loader_test,
    }
