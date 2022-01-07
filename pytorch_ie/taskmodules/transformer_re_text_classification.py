from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from pytorch_ie.data.document import BinaryRelation, Document, LabeledSpan
from pytorch_ie.taskmodules.taskmodule import (
    Metadata,
    TaskEncoding,
    TaskModule,
    BatchedModelOutput,
)

TransformerTextClassificationInputEncoding = Dict[str, Any]
TransformerTextClassificationTargetEncoding = List[int]
TransformerTextClassificationModelOutput = Dict[str, Any]


class TransformerRETextClassificationTaskModule(
    TaskModule[
        TransformerTextClassificationInputEncoding,
        TransformerTextClassificationTargetEncoding,
        TransformerTextClassificationModelOutput
    ]
):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        entity_annotation: str = "entities",
        relation_annotation: str = "relations",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        multi_label: bool = False,
        label_to_id: Optional[Dict[str, int]] = None,
        add_type_to_marker: bool = False,
        single_argument_pair: bool = True,
        append_markers: bool = False,
        entity_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            entity_annotation=entity_annotation,
            relation_annotation=relation_annotation,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            multi_label=multi_label,
            add_type_to_marker=add_type_to_marker,
            single_argument_pair=single_argument_pair,
            append_markers=append_markers,
            entity_labels=entity_labels,
        )

        self.entity_annotation = entity_annotation
        self.relation_annotation = relation_annotation
        self.padding = padding
        self.truncation = truncation
        self.label_to_id = label_to_id or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.multi_label = multi_label
        self.add_type_to_marker = add_type_to_marker
        self.single_argument_pair = single_argument_pair
        self.append_markers = append_markers
        self.entity_labels = entity_labels

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.argument_markers = None
        self.argument_markers_to_id = None
        self._create_argument_markers()

    def _create_argument_markers(self):
        argument_markers = {}
        for arg_type in ["head", "tail"]:
            is_head = arg_type == "head"

            for arg_pos in ["start", "end"]:
                is_start = arg_pos == "start"

                if self.add_type_to_marker:
                    for entity_type in self.entity_labels:
                        marker = (
                            f"[{'' if is_start else '/'}{'H' if is_head else 'T'}:{entity_type}]"
                        )
                        argument_markers[(arg_type, arg_pos, entity_type)] = marker
                else:
                    marker = f"[{'' if is_start else '/'}{'H' if is_head else 'T'}]"
                    argument_markers[(arg_type, arg_pos)] = marker

        self.tokenizer.add_tokens(list(argument_markers.values()), special_tokens=True)
        self.argument_markers = argument_markers
        self.argument_markers_to_id = {
            marker: self.tokenizer.vocab[marker] for marker in self.argument_markers.values()
        }

    def _config(self) -> Optional[Dict[str, Any]]:
        config = super()._config()
        config["label_to_id"] = self.label_to_id
        config["entity_labels"] = self.label_to_id
        return config

    def prepare(self, documents: List[Document]) -> None:
        entity_labels = set()
        relation_labels = set()
        for document in documents:
            entity_annotations = document.annotations(self.entity_annotation)
            relation_annotations = document.annotations(self.relation_annotation)

            if self.add_type_to_marker:
                for annotation in entity_annotations:
                    annotation_labels = (
                        annotation.label if annotation.is_multilabel else [annotation.label]
                    )
                    for label in annotation_labels:
                        if label not in entity_labels:
                            entity_labels.add(label)

            for annotation in relation_annotations:
                annotation_labels = (
                    annotation.label if annotation.is_multilabel else [annotation.label]
                )
                for label in annotation_labels:
                    if label not in relation_labels:
                        relation_labels.add(label)

        if "no_relation" in relation_labels:
            relation_labels.remove("no_relation")

        self.label_to_id["no_relation"] = 0
        current_id = 1
        for label in relation_labels:
            self.label_to_id[label] = current_id
            current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.entity_labels = entity_labels or []
        self._create_argument_markers()

    def _single_pair_insert_marker(
        self, documents: List[Document]
    ) -> Tuple[List[TransformerTextClassificationInputEncoding], Optional[List[Metadata]], Optional[List[Document]]]:
        input_encoding = []
        metadata = []
        new_documents = []

        for document in documents:
            entities = document.annotations(self.entity_annotation)

            encoding = self.tokenizer(
                document.text,
                padding=False,
                truncation=self.truncation,
                max_length=self.max_length,
                is_split_into_words=False,
                return_offsets_mapping=False,
            )

            relations: List[BinaryRelation] = document.annotations(self.relation_annotation)

            existing_head_tail = {(relation.head, relation.tail) for relation in relations}

            doc_metadata = {
                "head": [],
                "tail": [],
                "head_offset": [],
                "tail_offset": [],
            }

            head: LabeledSpan
            for head in entities:
                head_start = encoding.char_to_token(head.start)
                head_end = encoding.char_to_token(head.end - 1)

                if head_start is None or head_end is None:
                    continue

                tail: LabeledSpan
                for tail in entities:
                    assert not head.is_multilabel
                    assert not tail.is_multilabel

                    if head == tail:
                        continue

                    if relations and ((head, tail) not in existing_head_tail):
                        continue

                    tail_start = encoding.char_to_token(tail.start)
                    tail_end = encoding.char_to_token(tail.end - 1)

                    if tail_start is None or tail_end is None:
                        continue

                    if self.add_type_to_marker:
                        head_start_marker = self.argument_markers_to_id[
                            self.argument_markers[("head", "start", head.label)]
                        ]
                        head_end_marker = self.argument_markers_to_id[
                            self.argument_markers[("head", "end", head.label)]
                        ]
                        tail_start_marker = self.argument_markers_to_id[
                            self.argument_markers[("tail", "start", tail.label)]
                        ]
                        tail_end_marker = self.argument_markers_to_id[
                            self.argument_markers[("tail", "end", tail.label)]
                        ]
                    else:
                        head_start_marker = self.argument_markers_to_id[
                            self.argument_markers[("head", "start")]
                        ]
                        head_end_marker = self.argument_markers_to_id[
                            self.argument_markers[("head", "end")]
                        ]
                        tail_start_marker = self.argument_markers_to_id[
                            self.argument_markers[("tail", "start")]
                        ]
                        tail_end_marker = self.argument_markers_to_id[
                            self.argument_markers[("tail", "end")]
                        ]

                    head_items = (head_start, head_end + 1, head_start_marker, head_end_marker)
                    tail_items = (tail_start, tail_end + 1, tail_start_marker, tail_end_marker)

                    head_first = head_start < tail_start
                    first, second = (
                        (head_items, tail_items) if head_first else (tail_items, head_items)
                    )

                    first_start, first_end, first_start_marker, first_end_marker = first
                    second_start, second_end, second_start_marker, second_end_marker = second

                    input_ids = encoding["input_ids"]

                    first_tokens = input_ids[first_start:first_end]
                    second_tokens = input_ids[second_start:second_end]

                    new_input_ids = (
                        input_ids[:first_start]
                        + [first_start_marker]
                        + first_tokens
                        + [first_end_marker]
                        + input_ids[first_end:second_start]
                        + [second_start_marker]
                        + second_tokens
                        + [second_end_marker]
                        + input_ids[second_end:]
                    )

                    doc_metadata["head"].append(head)
                    doc_metadata["tail"].append(tail)

                    new_head_start = new_input_ids.index(head_start_marker)
                    new_head_end = new_input_ids.index(head_end_marker)
                    new_tail_start = new_input_ids.index(tail_start_marker)
                    new_tail_end = new_input_ids.index(tail_end_marker)

                    doc_metadata["head_offset"].append((new_head_start, new_head_end))
                    doc_metadata["tail_offset"].append((new_tail_start, new_tail_end))

                    input_encoding.append({"input_ids": new_input_ids})
                    new_documents.append(document)
                    metadata.append(doc_metadata)

                    doc_metadata = {
                        "head": [],
                        "tail": [],
                        "head_offset": [],
                        "tail_offset": [],
                    }

        return input_encoding, metadata, new_documents

    def encode_input(
        self, documents: List[Document]
    ) -> Tuple[List[TransformerTextClassificationInputEncoding], Optional[List[Metadata]], Optional[List[Document]]]:
        return self._single_pair_insert_marker(documents)

        # input_encoding = []
        # metadata = []
        # new_documents = []
        # for document in documents:
        #     entities = document.annotations(self.entity_annotation)

        #     encoding = self.tokenizer(
        #         document.text,
        #         padding=False,
        #         truncation=self.truncation,
        #         max_length=self.max_length,
        #         is_split_into_words=False,
        #         return_offsets_mapping=True,
        #         return_special_tokens_mask=True,
        #     )

        #     offset_mapping = encoding.pop("offset_mapping")

        #     doc_metadata = {
        #         "offset_mapping": offset_mapping,
        #         "head": [],
        #         "tail": [],
        #         "head_offset": [],
        #         "tail_offset": [],
        #     }
        #     for head in entities:
        #         head_start_idx = encoding.char_to_token(head.start)
        #         head_end_idx = encoding.char_to_token(head.end - 1)

        #         if head_start_idx is None or head_end_idx is None:
        #             continue

        #         for tail in entities:
        #             if head == tail:
        #                 continue

        #             tail_start_idx = encoding.char_to_token(tail.start)
        #             tail_end_idx = encoding.char_to_token(tail.end - 1)

        #             if tail_start_idx is None or tail_end_idx is None:
        #                 continue

        #             doc_metadata["head"].append(head)
        #             doc_metadata["tail"].append(tail)
        #             doc_metadata["head_offset"].append((head_start_idx, head_end_idx))
        #             doc_metadata["tail_offset"].append((tail_start_idx, tail_end_idx))

        #             if self.single_argument_pair:
        #                 input_encoding.append(encoding)
        #                 new_documents.append(document)
        #                 metadata.append(doc_metadata)

        #                 doc_metadata = {
        #                     "offset_mapping": offset_mapping,
        #                     "head": [],
        #                     "tail": [],
        #                     "head_offset": [],
        #                     "tail_offset": [],
        #                 }

        #     if not self.single_argument_pair:
        #         input_encoding.append(encoding)
        #         new_documents.append(document)
        #         metadata.append(doc_metadata)

        # return input_encoding, metadata, documents

    def encode_target(
        self,
        documents: List[Document],
        input_encodings: List[TransformerTextClassificationInputEncoding],
        metadata: Optional[List[Metadata]],
    ) -> List[TransformerTextClassificationTargetEncoding]:

        target: List[List[int]] = []
        for i, document in enumerate(documents):
            meta = metadata[i]

            relations: List[BinaryRelation] = document.annotations(self.relation_annotation)

            head_tail_to_label = {
                (relation.head, relation.tail): relation.label for relation in relations
            }

            for head, tail in zip(meta["head"], meta["tail"]):
                label = head_tail_to_label.get((head, tail), "no_relation")

                if self.multi_label:
                    raise NotImplementedError
                else:
                    label_ids = [self.label_to_id[label]]

            target.append(label_ids)

        return target

    def unbatch_output(self, output: BatchedModelOutput) -> List[TransformerTextClassificationModelOutput]:
        logits = output["logits"]

        output_label_probs = logits.sigmoid() if self.multi_label else logits.softmax(dim=-1)
        output_label_probs = output_label_probs.detach().cpu().numpy()

        decoded_output = []
        if self.multi_label:
            raise NotImplementedError
        else:
            result = {"labels": [], "probabilities": []}
            for batch_idx, label_id in enumerate(np.argmax(output_label_probs, axis=-1)):
                result["labels"].append(self.id_to_label[label_id])
                result["probabilities"].append(float(output_label_probs[batch_idx, label_id]))

            decoded_output.append(result)

        return decoded_output

    def create_annotations_from_output(
        self,
        output: TransformerTextClassificationModelOutput,
        encoding: TaskEncoding[TransformerTextClassificationInputEncoding, TransformerTextClassificationTargetEncoding],
    ) -> None:
        metadata = encoding.metadata
        labels = output["labels"]
        probabilities = output["probabilities"]
        heads = metadata["head"]
        tails = metadata["tail"]

        if self.multi_label:
            raise NotImplementedError

        else:
            for head, tail, label, probability in zip(heads, tails, labels, probabilities):
                if label != "no_relation":
                    yield (
                        self.relation_annotation,
                        BinaryRelation(
                            head=head,
                            tail=tail,
                            label=label,
                            score=probability,
                        ),
                    )

    def collate(
        self, encodings: List[TaskEncoding[TransformerTextClassificationInputEncoding, TransformerTextClassificationTargetEncoding]]
    ) -> tuple[Dict[str, Tensor], Optional[Tensor], list[Any], list[Document]]:

        input_features = [encoding.input for encoding in encodings]
        meta = [encoding.metadata for encoding in encodings]
        documents = [encoding.document for encoding in encodings]

        input_ = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if encodings[0].target is None:
            return input_, None, meta, documents

        target = [encoding.target for encoding in encodings]

        target = torch.tensor(target, dtype=torch.int64)

        if not self.multi_label:
            target = target.flatten()

        return input_, target, meta, documents
