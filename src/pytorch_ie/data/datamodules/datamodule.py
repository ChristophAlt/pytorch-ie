from typing import Any, Dict, Generic, List, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from pytorch_ie import Document
from pytorch_ie.taskmodules.taskmodule import (
    InputEncoding,
    TargetEncoding,
    TaskEncoding,
    TaskModule,
)


class TaskEncodingDataset(
    Dataset[TaskEncoding[InputEncoding, TargetEncoding]],
    Generic[InputEncoding, TargetEncoding],
):
    def __init__(self, encodings: List[TaskEncoding[InputEncoding, TargetEncoding]]):
        self._encodings = encodings

    def __getitem__(self, index) -> TaskEncoding[InputEncoding, TargetEncoding]:
        return self._encodings[index]

    def __len__(self):
        return len(self._encodings)


class DataModule(LightningDataModule, Generic[InputEncoding, TargetEncoding]):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        task_module: TaskModule[InputEncoding, TargetEncoding, Any, Any, Any],
        dataset: Dict[str, List[Document]],
        random_train_val_split: Optional[Tuple[int, int]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_config_path: Optional[str] = None,
        train_split: Optional[str] = "train",
        val_split: Optional[str] = "val",
        test_split: Optional[str] = "test",
        prepare_data_split: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.task_module = task_module
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.config_path = data_config_path
        self.dataset = dataset
        self.random_train_val_split = random_train_val_split
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        # per default, use train data to prepare the taskmodule
        self.prepare_data_split = prepare_data_split or self.train_split

        self._data: Dict[str, TaskEncodingDataset[InputEncoding, TargetEncoding]] = {}

    @property
    def num_train(self) -> int:
        if self.train_split is None:
            raise ValueError("no train_split assigned")
        data_train = self._data.get(self.train_split, None)
        if data_train is None:
            raise ValueError("can not get train size if setup() was not yet called")
        return len(data_train)

    def setup(self, stage: Optional[str] = None, **kwargs):

        if stage == "fit" or stage is None:
            self.task_module.prepare(self.dataset[self.prepare_data_split])

        for split in [self.train_split, self.val_split, self.test_split]:
            if split is None:
                continue
            self._data[split] = TaskEncodingDataset(
                self.task_module.encode(self.dataset[split], encode_target=True)
            )

        if self.random_train_val_split is not None:
            assert (
                self.train_split is not None
            ), "data_train has to be set to create random train dev splits from it"
            # type checking is broken for random_split, so we ignore it
            self._data[self.train_split], self._[self.val_split] = random_split(  # type: ignore
                self._data[self.train_split], self.random_train_val_split
            )

    def data_split(self, split: str) -> TaskEncodingDataset[InputEncoding, TargetEncoding]:
        if split is None or split not in self._data:
            raise ValueError(f"data for split={split} not available")
        return self._data[split]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_split(self.train_split),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_split(self.val_split),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_split(self.test_split),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.task_module.collate,
            shuffle=False,
        )
