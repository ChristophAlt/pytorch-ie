import dataclasses
import typing
from typing import Any, Dict, List, Optional, Set

from pytorch_ie.data.annotations import AnnotationList


def _depth_first_search(lst: List[str], visited: Set[str], graph: Dict[str, str], node: str):
    if node not in visited:
        lst.append(node)
        visited.add(node)
        neighbours = graph.get(node) or []
        for neighbour in neighbours:
            _depth_first_search(lst, visited, graph, neighbour)


def _get_annotation_fields(fields: List[dataclasses.Field]) -> Set[dataclasses.Field]:
    annotation_fields: Set[dataclasses.Field] = set()
    for field in fields:
        if typing.get_origin(field.type) is AnnotationList:
            annotation_fields.add(field)
    return annotation_fields


def annotation_field(target: Optional[str] = None):
    return dataclasses.field(metadata=dict(target=target), init=False, repr=False)


@dataclasses.dataclass
class Document:
    _annotation_targets: Dict[str, str] = dataclasses.field(default_factory=dict, init=False)

    def __post_init__(self):
        edges = set()
        for field in dataclasses.fields(self):
            if field.name == "_annotation_targets":
                continue

            field_origin = typing.get_origin(field.type)

            if field_origin is AnnotationList:
                annotation_target = field.metadata.get("target")
                edges.add((field.name, annotation_target))
                field_value = field.type(document=self, target=field.name)
                setattr(self, field.name, field_value)

        self._annotation_targets = {}
        for edge in edges:
            src, dst = edge
            if dst not in self._annotation_targets:
                self._annotation_targets[dst] = []
            self._annotation_targets[dst].append(src)

    def asdict(self):
        dct = {}
        for field in dataclasses.fields(self):
            if field.name == "_annotation_targets":
                continue

            value = getattr(self, field.name)

            if isinstance(value, AnnotationList):
                dct[field.name] = [v.asdict() for v in value]
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

        dependency_ordered_fields: List[dataclasses.Field] = []

        _depth_first_search(
            lst=dependency_ordered_fields,
            visited=set(),
            graph=doc._annotation_targets,
            node="text",
        )

        annotations = {}
        for field_name in dependency_ordered_fields:
            if field_name not in name_to_field:
                continue

            field = name_to_field[field_name]

            value = dct.get(field.name)

            if value is None or not value:
                continue

            # TODO: handle single annotations, e.g. a document-level label
            if typing.get_origin(field.type) is AnnotationList:
                annotation_class = typing.get_args(field.type)[0]
                for v in value:
                    v = dict(v)
                    annotation_id = v.pop("id")
                    annotations[annotation_id] = (
                        field.name,
                        annotation_class.fromdict(v, annotations),
                    )
            else:
                raise Exception("Error")

        for field_name, annotation in annotations.values():
            getattr(doc, field_name).append(annotation)

        return doc


@dataclasses.dataclass
class TextDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
