import dataclasses
from typing import Any, Dict, Optional, Tuple

from typing_extensions import TypeAlias

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, Document, annotation_field


@dataclasses.dataclass
class WithMetadata:
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class WithTokens:
    tokens: Tuple[str, ...]


@dataclasses.dataclass
class WithText:
    text: str


@dataclasses.dataclass
class TextBasedDocument(WithMetadata, WithText, Document):
    pass


@dataclasses.dataclass
class TokenBasedDocument(WithMetadata, WithTokens, Document):
    pass


# backwards compatibility
TextDocument: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class TextDocumentWithEntities(TextBasedDocument):
    entities: AnnotationList[Span] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithEntitiesAndRelations(TextDocumentWithEntities):
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithEntitiesRelationsAndLabeledPartitions(TextDocumentWithEntitiesAndRelations):
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesAndRelations(TextBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledEntities(TextDocumentWithLabeledEntitiesAndRelations):
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TextDocumentWithLabeledEntitiesRelationsAndLabeledPartitions(
    TextDocumentWithLabeledEntities
):
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")
