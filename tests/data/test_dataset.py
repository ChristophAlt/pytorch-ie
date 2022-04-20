import datasets
import pytest
from tests import FIXTURES_ROOT


def test_dataset(document_dataset):
    assert set(document_dataset.keys()) == {"train", "validation", "test"}

    assert len(document_dataset["train"]) == 8
    assert len(document_dataset["validation"]) == 2
    assert len(document_dataset["test"]) == 2

    train_doc5 = document_dataset["train"][4]
    assert train_doc5.id == "train_doc5"
    assert len(train_doc5.sentences) == 3
    assert len(train_doc5.entities) == 3
    assert len(train_doc5.relations) == 3

    assert train_doc5.sentences[1].text == "Entity G works at H."


def test_dataset_index(document_dataset):
    train_dataset = document_dataset["train"]
    assert train_dataset[4].id == "train_doc5"
    assert [doc.id for doc in train_dataset[0, 3, 5]] == ["train_doc1", "train_doc4", "train_doc6"]
    assert [doc.id for doc in train_dataset[2: 5]] ==  ["train_doc3", "train_doc4", "train_doc5"]


def test_dataset_map(document_dataset):
    train_dataset = document_dataset["train"]

    def clear_relations(document):
        document.relations.clear()
        return document

    assert sum([len(doc.relations) for doc in train_dataset]) == 7

    train_dataset1 = train_dataset.map(clear_relations)

    assert sum([len(doc.relations) for doc in train_dataset1]) == 0


def test_dataset_map_batched(document_dataset):
    train_dataset = document_dataset["train"]

    def clear_relations_batched(documents):
        assert len(documents) == 2
        for document in documents:
            document.relations.clear()
        return documents

    assert sum([len(doc.relations) for doc in train_dataset]) == 7

    train_dataset1 = train_dataset.map(clear_relations_batched, batched=True, batch_size=2)

    assert sum([len(doc.relations) for doc in train_dataset1]) == 0


import os
import tempfile
from typing import List, Optional
from unittest import TestCase

from datasets.builder import BuilderConfig
from datasets.load import dataset_module_factory, import_main_class
from datasets.utils.download_manager import GenerateMode
from datasets.utils.file_utils import DownloadConfig, is_remote_url
from datasets.utils.logging import get_logger
from datasets.utils.mock_download_manager import MockDownloadManager

logger = get_logger(__name__)


class DatasetTester:
    def __init__(self, parent):
        self.parent = parent if parent is not None else TestCase()

    def load_builder_class(self, dataset_name, is_local=False):
        # Download/copy dataset script
        if is_local is True:
            dataset_module = dataset_module_factory(os.path.join("datasets", dataset_name))
        else:
            dataset_module = dataset_module_factory(dataset_name, download_config=DownloadConfig(force_download=True))
        # Get dataset builder class
        builder_cls = import_main_class(dataset_module.module_path)
        return builder_cls

    def load_all_configs(self, dataset_name, is_local=False) -> List[Optional[BuilderConfig]]:
        # get builder class
        builder_cls = self.load_builder_class(dataset_name, is_local=is_local)
        builder = builder_cls

        if len(builder.BUILDER_CONFIGS) == 0:
            return [None]
        return builder.BUILDER_CONFIGS

    def check_load_dataset(self, dataset_name, configs, is_local=False, use_local_dummy_data=False):
        for config in configs:
            with tempfile.TemporaryDirectory() as processed_temp_dir, tempfile.TemporaryDirectory() as raw_temp_dir:

                # create config and dataset
                dataset_builder_cls = self.load_builder_class(dataset_name, is_local=is_local)
                name = config.name if config is not None else None
                dataset_builder = dataset_builder_cls(name=name, cache_dir=processed_temp_dir)

                # TODO: skip Beam datasets and datasets that lack dummy data for now
                if not dataset_builder.test_dummy_data:
                    logger.info("Skip tests for this dataset for now")
                    return

                if config is not None:
                    version = config.version
                else:
                    version = dataset_builder.VERSION

                def check_if_url_is_valid(url):
                    if is_remote_url(url) and "\\" in url:
                        raise ValueError(f"Bad remote url '{url} since it contains a backslash")

                # create mock data loader manager that has a special download_and_extract() method to download dummy data instead of real data
                mock_dl_manager = MockDownloadManager(
                    dataset_name=dataset_name,
                    config=config,
                    version=version,
                    cache_dir=raw_temp_dir,
                    use_local_dummy_data=use_local_dummy_data,
                    download_callbacks=[check_if_url_is_valid],
                )

                # packaged datasets like csv, text, json or pandas require some data files
                # builder_name = dataset_builder.__class__.__name__.lower()
                # if builder_name in _PACKAGED_DATASETS_MODULES:
                #     mock_dl_manager.download_dummy_data()
                #     path_to_dummy_data = mock_dl_manager.dummy_file
                #     dataset_builder.config.data_files = get_packaged_dataset_dummy_data_files(
                #         builder_name, path_to_dummy_data
                #     )
                #     for config_attr, value in get_packaged_dataset_config_attributes(builder_name).items():
                #         setattr(dataset_builder.config, config_attr, value)

                # mock size needed for dummy data instead of actual dataset
                if dataset_builder.info is not None:
                    # approximate upper bound of order of magnitude of dummy data files
                    one_mega_byte = 2 << 19
                    dataset_builder.info.size_in_bytes = 2 * one_mega_byte
                    dataset_builder.info.download_size = one_mega_byte
                    dataset_builder.info.dataset_size = one_mega_byte

                # generate examples from dummy data
                dataset_builder.download_and_prepare(
                    dl_manager=mock_dl_manager,
                    download_mode=GenerateMode.FORCE_REDOWNLOAD,
                    ignore_verifications=True,
                    try_from_hf_gcs=False,
                )

                # get dataset
                dataset = dataset_builder.as_dataset(ignore_verifications=True)

                # check that dataset is not empty
                self.parent.assertListEqual(sorted(dataset_builder.info.splits.keys()), sorted(dataset))
                for split in dataset_builder.info.splits.keys():
                    # check that loaded datset is not empty
                    self.parent.assertTrue(len(dataset[split]) > 0)

                # check that we can cast features for each task template
                task_templates = dataset_builder.info.task_templates
                if task_templates:
                    for task in task_templates:
                        task_features = {**task.input_schema, **task.label_schema}
                        for split in dataset:
                            casted_dataset = dataset[split].prepare_for_task(task)
                            self.parent.assertDictEqual(task_features, casted_dataset.features)
                            del casted_dataset
                del dataset


def test_with_tester():
    dataset_name = FIXTURES_ROOT / "datasets" / "conll2003"

    dataset_tester = DatasetTester(parent=None)
    configs = dataset_tester.load_all_configs(dataset_name, is_local=True)[:1]
    dataset_tester.check_load_dataset(dataset_name=dataset_name, configs=configs, is_local=True, use_local_dummy_data=True)


@pytest.mark.slow
def test_load_with_hf_datasets():
    dataset_dir = FIXTURES_ROOT / "datasets" / "conll2003"

    dataset = datasets.load_dataset(
        path=str(dataset_dir / "conll2003.py"),
    )

    assert set(dataset.keys()) == {"train", "validation", "test"}

    # TODO: the updated CoNLL03 data files have two newlines at the end
    # this results in one additional example in train, validation, and test
    assert len(dataset["train"]) == 14042  # 14041
    assert len(dataset["validation"]) == 3251
    assert len(dataset["test"]) == 3454

    train_doc1 = dataset["train"][0]
