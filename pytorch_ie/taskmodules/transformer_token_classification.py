import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy

from pytorch_ie.data.document import Annotation, Document, LabeledSpan
from pytorch_ie.data.span_utils import bio_tags_to_spans
from pytorch_ie.models.transformer_token_classification import (
    TransformerTokenClassificationModelBatchOutput,
    TransformerTokenClassificationModelStepBatchEncoding,
)
from pytorch_ie.taskmodules.taskmodule import Metadata, TaskEncoding, TaskModule

"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""
TransformerTokenClassificationInputEncoding = BatchEncoding
TransformerTokenClassificationTargetEncoding = List[int]

TransformerTokenClassificationTaskEncoding = TaskEncoding[
    TransformerTokenClassificationInputEncoding, TransformerTokenClassificationTargetEncoding
]
TransformerTokenClassificationTaskOutput = Dict[str, Any]

_TransformerTokenClassificationTaskModule = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    TransformerTokenClassificationInputEncoding,
    TransformerTokenClassificationTargetEncoding,
    TransformerTokenClassificationModelStepBatchEncoding,
    TransformerTokenClassificationModelBatchOutput,
    TransformerTokenClassificationTaskOutput,
]

logger = logging.getLogger(__name__)


class TransformerTokenClassificationTaskModule(_TransformerTokenClassificationTaskModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        partition_annotation: Optional[str] = None,
        single_sentence: bool = False,  # deprecated, set partition_annotation instead
        sentence_annotation: str = "sentences",  # deprecated, use partition_annotation instead
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        label_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.entity_annotation = entity_annotation
        self.partition_annotation = partition_annotation
        # backwards compatibility
        if single_sentence:
            self.partition_annotation = sentence_annotation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def _config(self) -> Dict[str, Any]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        labels = set()
        for document in documents:
            entities = document.span_annotations(self.entity_annotation)
            assert (
                entities is not None
            ), f"document has no span annotations with name '{self.entity_annotation}'"

            for entity in entities:
                labels.update(entity.labels)

        self.label_to_id["O"] = 0
        current_id = 1
        for label in sorted(labels):
            for prefix in ["B", "I"]:
                self.label_to_id[f"{prefix}-{label}"] = current_id
                current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[
        List[TransformerTokenClassificationInputEncoding],
        List[Metadata],
        Optional[List[Document]],
    ]:
        metadata = []
        expanded_documents = []
        input_ = []
        for doc in documents:
            if self.partition_annotation is not None:
                partitions_or_none = doc.span_annotations(self.partition_annotation)
                assert (
                    partitions_or_none
                ), f"document has no span annotations with name '{self.partition_annotation}'"
                partitions = partitions_or_none
            else:
                partitions = [LabeledSpan(start=0, end=len(doc.text), label="FULL_DOCUMENT")]

            for partition_index, partition in enumerate(partitions):
                encoding = self.tokenizer(
                    doc.text[partition.start : partition.end],
                    padding=False,
                    truncation=False,
                    max_length=None,
                    is_split_into_words=False,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                current_metadata = {
                    "offset_mapping": encoding.pop("offset_mapping"),
                    "special_tokens_mask": encoding.pop("special_tokens_mask"),
                }
                if self.partition_annotation is not None:
                    current_metadata["sentence_index"] = partition_index
                metadata.append(current_metadata)
                input_.append(encoding)
                expanded_documents.append(doc)

        return input_, metadata, expanded_documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerTokenClassificationInputEncoding],
        metadata: List[Metadata],
    ) -> List[TransformerTokenClassificationTargetEncoding]:
        target = []
        if self.partition_annotation is not None:
            for i, document in enumerate(documents):
                entities = document.span_annotations(self.entity_annotation)
                assert (
                    entities
                ), f"document has no span annotations with name '{self.entity_annotation}'"
                sentence_idx = metadata[i]["sentence_index"]
                partitions = document.span_annotations(self.partition_annotation)
                assert (
                    partitions
                ), f"document has no span annotations with name '{self.partition_annotation}'"
                sentence = partitions[sentence_idx]

                word_ids = input_encodings[i].word_ids()
                label_ids = [
                    self.label_pad_token_id if word_ids[j] is None else self.label_to_id["O"]
                    for j in range(len(word_ids))
                ]

                entity: LabeledSpan
                for entity in entities:
                    if entity.start < sentence.start or entity.end > sentence.end:
                        continue

                    entity_start = entity.start - sentence.start
                    entity_end = entity.end - sentence.start

                    start_idx = input_encodings[i].char_to_token(entity_start)
                    end_idx = input_encodings[i].char_to_token(entity_end - 1)
                    # TODO: remove this is if case
                    if start_idx is None or end_idx is None:
                        logger.warning(
                            f"Entity annotation does not start or end with a token, it will be skipped: {entity}"
                        )
                        continue

                    for j in range(start_idx, end_idx + 1):
                        prefix = "B" if j == start_idx else "I"
                        label_ids[j] = self.label_to_id[f"{prefix}-{entity.label_single}"]

                target.append(label_ids)
        else:
            for i, document in enumerate(documents):
                word_ids = input_encodings[i].word_ids()
                label_ids = [
                    self.label_pad_token_id if word_ids[j] is None else self.label_to_id["O"]
                    for j in range(len(word_ids))
                ]

                entities = document.span_annotations(self.entity_annotation)
                assert (
                    entities
                ), f"document has no span annotations with name '{self.entity_annotation}'"

                for entity in entities:
                    start_idx = input_encodings[i].char_to_token(entity.start)
                    end_idx = input_encodings[i].char_to_token(entity.end - 1)
                    for j in range(start_idx, end_idx + 1):
                        prefix = "B" if j == start_idx else "I"
                        label_ids[j] = self.label_to_id[f"{prefix}-{entity.label_single}"]

                target.append(label_ids)

        return target

    def unbatch_output(
        self, output: TransformerTokenClassificationModelBatchOutput
    ) -> Sequence[TransformerTokenClassificationTaskOutput]:
        logits = output["logits"]
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
        indices = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        tags = [[self.id_to_label[e] for e in b] for b in indices]
        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        encoding: TransformerTokenClassificationTaskEncoding,
        output: TransformerTokenClassificationTaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        if self.partition_annotation is not None:
            document = encoding.document
            metadata = encoding.metadata
            partitions = document.span_annotations(self.partition_annotation)
            assert (
                partitions
            ), f"document has no span annotations with name '{self.partition_annotation}'"
            sentence = partitions[metadata["sentence_index"]]

            tag_sequence = [
                "O" if stm else tag
                for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
            ]

            spans = bio_tags_to_spans(tag_sequence)
            for label, (start, end) in spans:
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        sentence.start + metadata["offset_mapping"][start][0],
                        sentence.start + metadata["offset_mapping"][end][1],
                        label,
                    ),
                )
        else:
            metadata = encoding.metadata

            tag_sequence = [
                "O" if stm else tag
                for tag, stm in zip(output["tags"], metadata["special_tokens_mask"])
            ]

            spans = bio_tags_to_spans(tag_sequence)
            for label, (start, end) in spans:
                yield (
                    self.entity_annotation,
                    LabeledSpan(
                        metadata["offset_mapping"][start][0],
                        metadata["offset_mapping"][end][1],
                        label,
                    ),
                )

    def collate(
        self, encodings: List[TransformerTokenClassificationTaskEncoding]
    ) -> TransformerTokenClassificationModelStepBatchEncoding:
        input_features = [encoding.input for encoding in encodings]

        input_ = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not encodings[0].has_target:
            return input_, None

        target_list: List[TransformerTokenClassificationTargetEncoding] = [
            encoding.target for encoding in encodings
        ]

        sequence_length = torch.tensor(input_["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            target_list_padded = [
                list(t) + [self.label_pad_token_id] * (sequence_length - len(t))
                for t in target_list
            ]
        else:
            target_list_padded = [
                [self.label_pad_token_id] * (sequence_length - len(t)) + list(t)
                for t in target_list
            ]

        input_ = {k: torch.tensor(v, dtype=torch.int64) for k, v in input_.items()}
        target = torch.tensor(target_list_padded, dtype=torch.int64)

        return input_, target
