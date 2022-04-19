import dataclasses
from dataclasses import dataclass, field
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_ie.models import TransformerTokenClassificationModel
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule

from typing import List, Optional, get_type_hints, Union, Callable, Mapping

from tests import FIXTURES_ROOT

from datasets import load_dataset
import datasets

# from pytorch_ie.data.document import LabeledSpan, BinaryRelation

from datasets.load import load_dataset_builder

import abc
import warnings

from datasets.load import extend_dataset_builder_for_streaming
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES


class GeneratorBasedBuilder(datasets.builder.GeneratorBasedBuilder):
    # Default batch size used by the ArrowWriter
    # It defines the number of samples that are kept in memory before writing them
    # and also the length of the arrow chunks
    # None means that the ArrowWriter will use its default value
    DOCUMENT_TYPE = None

    BASE_PATH = None

    def __init__(self, **kwargs):
        builder_kwargs = dict(kwargs)
        builder_kwargs.pop("hash", None)
        builder_kwargs.pop("base_path", None)
        self.base_builder = load_dataset_builder(
            path=self.BASE_PATH,
            **builder_kwargs,
        )
        super().__init__(**kwargs)

    def _info(self):
        return self.base_builder._info()
    
    def _split_generators(self, dl_manager):
        return self.base_builder._split_generators(dl_manager)

    def _generate_examples(self, filepath):
        return self.base_builder._generate_examples(filepath)

    @abc.abstractmethod
    def _generate_document(self, example, dataset):
        pass

    def _generate_document_kwargs(self, dataset):
        return None

    def _post_process(self, dataset: datasets.Dataset, resources_paths: Mapping[str, str]) -> Optional[datasets.Dataset]:
        fn_kwargs = {}
        additional_kwargs = self._generate_document_kwargs(dataset)

        if additional_kwargs is not None:
            fn_kwargs.update(additional_kwargs)

        def decorate(f):
            """
            Decorate the mapped function, so that its first argument is wrapped with a LazyDict to be used internally
            but a standard dictionary is returned at the end of the mapping.
            """
            import pandas as pd
            

            @wraps(f)
            def decorated(item, *args, **kwargs):
                if isinstance(item, list):
                    # return [e.asdict() for e in f(item)]
                    return pd.DataFrame([e.asdict() for e in f(item, *args, **kwargs)]).to_dict(orient="list")
                else:
                    return f(item, *args, **kwargs).asdict()
            
            return decorated
        
        mapped_dataset = dataset.map(decorate(self._generate_document), fn_kwargs=fn_kwargs)

        document_dataset = Dataset.from_hf_dataset(mapped_dataset)
        document_dataset.set_format("document", document_type=self.DOCUMENT_TYPE)

        return document_dataset


def dfs(lst, visited, graph, node):
    if node not in visited:
        lst.append(node)
        visited.add(node)
        neighbours = graph.get(node) or []
        for neighbour in neighbours:
            dfs(lst, visited, graph, neighbour)

# TODO: metadata is missing
@dataclass
class Document:
    text: str
    id: Optional[str] = None

    def __post_init__(self):
        edges = set()
        for name, field_ in self.__dataclass_fields__.items():
            if name in {"text", "id"}:
                continue

            target = field_.metadata.get("target")
            edges.add((name, target))

            setattr(self, name, AnnotationList(self, target))

        self._targets = {}
        for edge in edges:
            src, dst = edge
            if dst not in self._targets:
                self._targets[dst] = []
            self._targets[dst].append(src)

    def asdict(self):
        dct = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in {"text", "id"}:
                dct[f.name] = value
            else:
                if isinstance(value, (list, tuple)):
                    dct[f.name] = list(type(value)(v.asdict() for v in value))
                elif isinstance(value, AnnotationList):
                    dct[f.name] = [v.asdict() for v in value]
                else:
                    raise Exception("Error")
        return dct

    @classmethod
    def fromdict(cls, dct):
        doc = cls(text=dct["text"], id=dct.get("id"))

        annotation_fields = {f.name: f for f in dataclasses.fields(cls) if f.name not in {"id", "text"}}

        flds = []

        dfs(flds, set(), doc._targets, "text")

        annotations = {}
        for field_name in flds:
            if field_name not in annotation_fields:
                continue

            f = annotation_fields[field_name]

            value = dct.get(f.name)

            if value is None or not value:
                continue

            # TODO: this seems a bit hacky
            # also, how to handle single annotations, e.g. a document-level label
            if isinstance(value, (list, tuple, AnnotationList)):
                annotation_class = get_type_hints(cls)[f.name].__args__[0]
                for v in value:
                    v = dict(v)
                    annotation_id = v.pop("id")
                    annotations[annotation_id] = (f.name, annotation_class.fromdict(v, annotations))
            else:
                raise Exception("Error")

        for field_name, annotation in annotations.values():
            getattr(doc, field_name).append(annotation)

        return doc

# @dataclass(eq=True, frozen=True)
# class AnnotationBase:
#     doc: Optional[Document] = field(default=None, init=False, repr=False, hash=False, compare=False)

@dataclass(eq=True, frozen=True)
class AnnotationBase:
    _target: Optional[Union["AnnotationBase", str]] = field(default=None, init=False, repr=False, hash=False, compare=False)

    def set_target(self, value):
        object.__setattr__(self, '_target', value)
        # self._target = target

    def target(self):
        return self._target

    def asdict(self):
        dct = dataclasses.asdict(self)
        dct["id"] = hash(self)
        del dct["_target"]
        return dct

    @classmethod
    def fromdict(cls, dct, annotations=None):
        return cls(**dct)

@dataclass(eq=True, frozen=True)
class Span(AnnotationBase):
    start: int
    end: int

    def text(self):
        return self._target[self.start: self.end]

@dataclass(eq=True, frozen=True)
class LabeledSpan(Span):
    label: str
    score: float = 1.0

@dataclass(eq=True, frozen=True)
class BinaryRelation(AnnotationBase):
    head: Span
    tail: Span
    label: str
    score: float = 1.0

    def asdict(self):
        dct = super().asdict()
        # dct = dataclasses.asdict(self)
        # dct["id"] = hash(self)
        dct["head"] = hash(self.head)
        dct["tail"] = hash(self.tail)
        return dct

    @classmethod
    def fromdict(cls, dct, annotations=None):
        if annotations is not None:
            head_id = dct["head"]
            tail_id = dct["tail"]

            dct["head"] = annotations[head_id][1]
            dct["tail"] = annotations[tail_id][1]

        return cls(**dct)

@dataclass(eq=True, frozen=True)
class Label(AnnotationBase):
    label: str
    score: float = 1.0

from collections.abc import Sequence

class AnnotationList(Sequence):
    def __init__(self, document, target):
        self._document = document
        self._target = target
        self._annotations = []

    # TODO: not sure this is a good idea
    def __eq__(self, other: object) -> bool:
        return self._target == other._target and self._annotations == other._annotations

    def __getitem__(self, idx):
        return self._annotations[idx]

    def __len__(self):
        return len(self._annotations)

    def append(self, annotation: AnnotationBase):
        annotation.set_target(getattr(self._document, self._target))
        self._annotations.append(annotation)

    def __repr__(self) -> str:
        return f"AnnotationList({str(self._annotations)})"
    
    def clear(self):
        for annotation in self._annotations:
            annotation.set_target(None)
        self._annotations = []

def annotation_field(target: Optional[str] = None):
    return field(metadata=dict(target=target), init=False, repr=False)

@dataclass
class MyDocument(Document):
    sentences: AnnotationList[Span] = annotation_field(target="text")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    # TODO: how to handle this case?
    # topic: Annotation[Label] = ???


def test_create_static_document():
    # document = MyDocument(text="Entity A works at B.")
    document = MyDocument(text="Entity A works at B.", id="ABC123")

    sentence1 = Span(start=0, end=20)

    entity1 = LabeledSpan(start=0, end=8, label="PER")
    entity2 = LabeledSpan(start=18, end=19, label="ORG")

    relation1 = BinaryRelation(head=entity1, tail=entity2, label="per:employee_of")

    # this should fail because head and tail entity annotations are not in the document
    # document.relations.append(relation1)

    # this should work
    document.entities.append(entity1)
    document.entities.append(entity2)
    document.relations.append(relation1)

    print(document.asdict())

    print(MyDocument.fromdict(document.asdict()))

    assert MyDocument.fromdict(document.asdict()) == document
    assert MyDocument.fromdict(document.asdict()).asdict() == document.asdict()

    # assert document.asdict() == {
    #     "text": "Entity A works at B.",
    #     "id": "ABC123",
    #     "sentences": [
    #         {"start": 0, "end": 20, "id": 1}
    #     ],
    #     "entities": [
    #         {"start": 0, "end": 8, "label": "PER", "id": 2},
    #         {"start": 18, "end": 19, "label": "ORG", "id": 3},
    #     ],
    #     "relations": [
    #         {"head": 2, "tail": 3, "label": "per:employee_of", "id": 4}
    #     ],
    # }

    # I'm not sure what to do in this case (because entity1 is referenced by relation1)
    # document.entities.remove(entity1)

    # this should work
    # document.relations.remove(relation1)
    # document.entities.remove(entity1)


from datasets.formatting import _register_formatter
from datasets.formatting.formatting import Formatter
import pyarrow as pa
from functools import wraps


class DocumentFormatter(Formatter[dict, list, dict]):
    def __init__(self, document_type, features=None, decoded=True, **kwargs):
        super().__init__(features=None, decoded=None)
        self.document_type = document_type

    def format_row(self, pa_table: pa.Table) -> dict:
        row = self.python_arrow_extractor().extract_row(pa_table)
        return self.document_type.fromdict(row)

    def format_column(self, pa_table: pa.Table) -> list:
        return []

    def format_batch(self, pa_table: pa.Table) -> dict:
        batch = self.simple_arrow_extractor().extract_batch(pa_table).to_pylist()
        return [self.document_type.fromdict(b) for b in batch]

_register_formatter(DocumentFormatter, "document")


class Dataset(datasets.Dataset):
    @classmethod
    def from_hf_dataset(cls, dataset: datasets.Dataset):
        return cls(
            arrow_table=dataset._data,
            info=dataset.info,
            split=dataset.split,
            indices_table=dataset._indices,
            fingerprint=dataset._fingerprint,
        )
    
    def map(
        self,
        function: Optional[Callable] = None,
        as_documents: bool = False,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[datasets.Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> "Dataset":

        def decorate(f):
            """
            Decorate the mapped function, so that its first argument is wrapped with a LazyDict to be used internally
            but a standard dictionary is returned at the end of the mapping.
            """
            import pandas as pd
            

            @wraps(f)
            def decorated(item, *args, **kwargs):
                if isinstance(item, list):
                    # return [e.asdict() for e in f(item)]
                    return pd.DataFrame([e.asdict() for e in f(item, *args, **kwargs)]).to_dict(orient="list")
                else:
                    return f(item, *args, **kwargs).asdict()
            
            return decorated
        
        dataset = super().map(
            function = decorate(function) if as_documents else function,
            with_indices = with_indices,
            with_rank = with_rank,
            input_columns = input_columns,
            batched = batched,
            batch_size = batch_size,
            drop_last_batch = drop_last_batch,
            remove_columns = remove_columns,
            keep_in_memory = keep_in_memory,
            load_from_cache_file = load_from_cache_file,
            cache_file_name = cache_file_name,
            writer_batch_size = writer_batch_size,
            features = features,
            disable_nullable = disable_nullable,
            fn_kwargs = fn_kwargs,
            num_proc = num_proc,
            suffix_template = suffix_template,
            new_fingerprint = new_fingerprint,
            desc = desc,
        )

        return Dataset(
            arrow_table=dataset._data,
            info=dataset.info,
            split=dataset.split,
            indices_table=dataset._indices,
            fingerprint=dataset._fingerprint,
        )


def test_load_with_datasets():
    dataset_dir = FIXTURES_ROOT / "datasets" / "json_2"

    dataset = load_dataset(
        # path="json",
        path=str(FIXTURES_ROOT / "datasets" / "json_2" / "json2.py"),
        field="data",
        data_files={
            "train": str(dataset_dir / "train.json"),
            "validation": str(dataset_dir / "val.json"),
            "test": str(dataset_dir / "test.json"),
        }
    )

    print(dataset)

    def convert_to_doc_dict(example):
        doc = MyDocument(text=example["text"], id=example["id"])

        sentences = [Span.fromdict(dct) for dct in example["sentences"]]
        entities = [LabeledSpan.fromdict(dct) for dct in example["entities"]]
        relations = [BinaryRelation(head=entities[rel["head"]], tail=entities[rel["tail"]], label=rel["label"]) for rel in example["relations"]]

        for sentence in sentences:
            doc.sentences.append(sentence)

        for entity in entities:
            doc.entities.append(entity)

        for relation in relations:
            doc.relations.append(relation)

        # this should be done transparently (but conceptionally it works)
        return doc.asdict()

    train_dataset = dataset["train"].map(convert_to_doc_dict)

    # train_dataset.set_transform(MyDocument.fromdict)

    train_dataset.set_format("document", document_type=MyDocument)

    print(train_dataset[0])

    print(train_dataset[0:2])

    print(train_dataset[0, 1, 2])

    def some_func(document):
        print(document.asdict())
        return document

    train_dataset2 = train_dataset.map(some_func, as_documents=True)
    train_dataset2.set_format("document", document_type=MyDocument)

    print(train_dataset2)

    def some_func_batched(documents):
        for doc in documents:
            doc.relations.clear()
        return documents

    train_dataset3 = train_dataset2.map(some_func_batched, as_documents=True, batch_size=2, batched=True)
    train_dataset3.set_format("document", document_type=MyDocument)

    print(train_dataset3)


def test_load_with_datasets_conll2003():
    dataset = load_dataset(
        path=str(FIXTURES_ROOT / "datasets" / "conll2003.py"),
    )

    print(dataset)

# def test_load_dataset_and_train():
#     pass
    # pl.seed_everything(42)

    # model_name = "bert-base-cased"
    # num_epochs = 1
    # batch_size = 32

    # task_module = TransformerTokenClassificationTaskModule(
    #     tokenizer_name_or_path=model_name,
    #     max_length=128,
    # )

    # model = TransformerTokenClassificationModel(
    #     model_name_or_path=model_name,
    #     num_classes=len(task_module.label_to_id),
    #     learning_rate=1e-4,
    # )

    # document_dataset is cached in document format (serialized as a dict of primitive types)
    # TODO: create our own load_dataset function (same interface as HF load_dataset, and mostly same logic)
    # - ideally we'd have our own custom loading scripts that use the original CoNLL03 dataset loading script
    #   provided by HF datasets and only extend it with functionality to convert it to a document
    # - the challenge is to find a generic way that allows us to serialize and deserialize a Document.
    #   the main problem is that annotations target specific fields, e.g. entities target text, relations target entities,
    #   and this must be encoded during serialization, or stored alongside the serialized data
    # - this leads to another challenge: how to handle the schema definition when using single documents and a pipeline
    # document_dataset = load_dataset(path="conll2003", name="...")

    # pie.schema.Document(
    #     {
    #         "id": pie.schema.Value("string"),
    #         "text": pie.schema.Value("string"),
    #         "sentences": pie.schema.Sequence(pie.schema.Span(target="text")),
    #         "entities": pie.schema.Sequence(pie.schema.LabeledSpan(target="text", class_labels=...)),
    #         "relations": pie.schema.Sequence(pie.schema.BinaryRelation(target="entities", class_labels=...)),
    #     }
    # )

    # document_dataset is a Dataset or DatasetDict, or an IterableDataset or IterableDatasetDict
    # document_1 = document_dataset["train"][0]

    # this may be an option but I don't think it's worth the effort
    # train_dataset = document_dataset["train"]
    # train_dataset.map(lambda documents: task_module.encode(documents, encode_target=True), batched=True)

    # train_documents = document_dataset["train"]
    # # TODO: TaskModule.encode should be able to work with iterables as well
    # train_dataset = task_module.encode(train_documents, encode_target=True)

    # train_dataloader = DataLoader(
    #     train_dataset,  # type: ignore
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=task_module.collate,
    # )

    # trainer = pl.Trainer(
    #     fast_dev_run=False,
    #     max_epochs=num_epochs,
    #     gpus=0,
    #     checkpoint_callback=False,
    #     precision=32,
    # )
    # trainer.fit(model, train_dataloader)
