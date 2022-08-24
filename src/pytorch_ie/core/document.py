import dataclasses
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union, overload


def _enumerate_dependencies(
    resolved: List[str],
    dependency_graph: Dict[str, List[str]],
    nodes: List[str],
    current_path: Optional[Set[str]] = None,
):
    current_path = current_path or set()
    for node in nodes:
        if node in current_path:
            raise ValueError(f"circular dependency detected at node: {node}")
        if node not in resolved:
            # terminal nodes
            if node not in dependency_graph:
                resolved.append(node)
            # nodes with dependencies
            else:
                # enumerate all dependencies first, then append itself
                _enumerate_dependencies(
                    resolved,
                    dependency_graph,
                    nodes=dependency_graph[node],
                    current_path=current_path | {node},
                )
                resolved.append(node)


def _get_annotation_fields(fields: List[dataclasses.Field]) -> Set[dataclasses.Field]:
    return {field for field in fields if typing.get_origin(field.type) is AnnotationList}


def annotation_field(target: Optional[str] = None):
    return dataclasses.field(metadata=dict(target=target), init=False, repr=False)


@dataclasses.dataclass(eq=True, frozen=True)
class Annotation:
    _target: Optional[Union["AnnotationList", str]] = dataclasses.field(
        default=None, init=False, repr=False, hash=False
    )

    def set_target(self, value: Union["AnnotationList", str, None]):
        object.__setattr__(self, "_target", value)

    @property
    def target(self) -> Optional[Union["AnnotationList", str]]:
        return self._target

    def asdict(self) -> Dict[str, Any]:
        dct = dataclasses.asdict(self)
        dct["_id"] = hash(self)
        del dct["_target"]
        return dct

    @classmethod
    def fromdict(
        cls,
        dct: Dict[str, Any],
        annotation_store: Optional[Dict[int, Tuple[str, "Annotation"]]] = None,
    ):
        tmp_dct = dict(dct)
        tmp_dct.pop("_id", None)
        return cls(**tmp_dct)


T = TypeVar("T", covariant=False, bound="Annotation")


class BaseAnnotationList(Sequence[T]):
    def __init__(self, document: "Document", target: "str"):
        self._document = document
        self._target = target
        self._annotations: List[T] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseAnnotationList):
            return NotImplemented

        return self._target == other._target and self._annotations == other._annotations

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice) -> List[T]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        return self._annotations[index]

    def __len__(self) -> int:
        return len(self._annotations)

    def append(self, annotation: T) -> None:
        target = getattr(self._document, self._target) if self._target is not None else None
        annotation.set_target(target)
        self._annotations.append(annotation)

    def extend(self, annotations: Iterable[T]) -> None:
        for annotation in annotations:
            self.append(annotation)

    def __repr__(self) -> str:
        return f"BaseAnnotationList({str(self._annotations)})"

    def clear(self):
        for annotation in self._annotations:
            annotation.set_target(None)
        self._annotations = []


class AnnotationList(BaseAnnotationList[T]):
    def __init__(self, document: "Document", target: "str"):
        super().__init__(document=document, target=target)
        self._predictions: BaseAnnotationList[T] = BaseAnnotationList(document, target)

    @property
    def predictions(self) -> BaseAnnotationList[T]:
        return self._predictions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnotationList):
            return NotImplemented

        return super().__eq__(other) and self.predictions == other.predictions

    def __repr__(self) -> str:
        return f"AnnotationList({str(self._annotations)})"


D = TypeVar("D", bound="Document")


@dataclasses.dataclass
class Document(Mapping[str, Any]):
    _annotation_graph: Dict[str, List[str]] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )
    _annotation_fields: Set[str] = dataclasses.field(default_factory=set, init=False, repr=False)

    @classmethod
    def fields(cls):
        return [
            f
            for f in dataclasses.fields(cls)
            if f.name not in {"_annotation_graph", "_annotation_fields"}
        ]

    @classmethod
    def annotation_fields(cls):
        return _get_annotation_fields(list(dataclasses.fields(cls)))

    def __getitem__(self, key: str) -> AnnotationList:
        if key not in self._annotation_fields:
            raise KeyError(f"Document has no attribute '{key}'.")
        return getattr(self, key)

    def __iter__(self):
        return iter(self._annotation_fields)

    def __len__(self):
        return len(self._annotation_fields)

    def __post_init__(self):
        targeted = set()
        for field in dataclasses.fields(self):
            if field.name == "_annotation_graph":
                continue

            field_origin = typing.get_origin(field.type)

            if field_origin is AnnotationList:
                self._annotation_fields.add(field.name)

                annotation_target = field.metadata.get("target")
                if annotation_target is not None:
                    targeted.add(annotation_target)
                    if field.name not in self._annotation_graph:
                        self._annotation_graph[field.name] = []
                    self._annotation_graph[field.name].append(annotation_target)

                field_value = field.type(document=self, target=annotation_target)
                setattr(self, field.name, field_value)

        if "_artificial_root" in self._annotation_graph:
            raise ValueError(
                "the annotation graph already contains a node _artificial_root, this is not allowed"
            )
        self._annotation_graph["_artificial_root"] = list(self._annotation_fields - targeted)

    def asdict(self):
        dct = {}
        for field in self.fields():

            value = getattr(self, field.name)

            if isinstance(value, AnnotationList):
                dct[field.name] = {
                    "annotations": [v.asdict() for v in value],
                    "predictions": [v.asdict() for v in value.predictions],
                }
            elif isinstance(value, dict):
                dct[field.name] = value or None
            else:
                dct[field.name] = value

        return dct

    @classmethod
    def fromdict(cls, dct):
        fields = dataclasses.fields(cls)
        annotation_fields = _get_annotation_fields(fields)

        cls_kwargs = {}
        for field in fields:
            if field not in annotation_fields:
                value = dct.get(field.name)

                if value is not None:
                    cls_kwargs[field.name] = value

        doc = cls(**cls_kwargs)

        name_to_field = {f.name: f for f in annotation_fields}

        dependency_ordered_fields: List[str] = []
        _enumerate_dependencies(
            dependency_ordered_fields,
            dependency_graph=doc._annotation_graph,
            nodes=doc._annotation_graph["_artificial_root"],
        )

        annotations = {}
        predictions = {}
        for field_name in dependency_ordered_fields:
            # terminal nodes do not have to be an annotation field (e.g. the text field)
            if field_name not in name_to_field:
                continue

            field = name_to_field[field_name]

            value = dct.get(field.name)

            if value is None or not value:
                continue

            # TODO: handle single annotations, e.g. a document-level label
            if typing.get_origin(field.type) is AnnotationList:
                annotation_class = typing.get_args(field.type)[0]
                # build annotations
                for annotation_data in value["annotations"]:
                    annotation_dict = dict(annotation_data)
                    annotation_id = annotation_dict.pop("_id")
                    annotations[annotation_id] = (
                        field.name,
                        # annotations can only reference annotations
                        annotation_class.fromdict(annotation_dict, annotations),
                    )
                # build predictions
                for annotation_data in value["predictions"]:
                    annotation_dict = dict(annotation_data)
                    annotation_id = annotation_dict.pop("_id")
                    predictions[annotation_id] = (
                        field.name,
                        # predictions can reference annotations and predictions
                        annotation_class.fromdict(annotation_dict, {**annotations, **predictions}),
                    )
            else:
                raise Exception("Error")

        for field_name, annotation in annotations.values():
            getattr(doc, field_name).append(annotation)

        for field_name, annotation in predictions.values():
            getattr(doc, field_name).predictions.append(annotation)

        return doc

    def as_type(self, new_type: typing.Type[D], field_mapping: Optional[Dict[str, str]] = None):
        field_mapping = field_mapping or {}
        new_doc = new_type.fromdict({field_mapping.get(k, k): v for k, v in self.asdict().items()})
        return new_doc
