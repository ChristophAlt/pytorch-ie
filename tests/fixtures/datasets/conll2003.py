import tests.data.test_new_document_and_datasets
from tests.data.test_new_document_and_datasets import Document, AnnotationList, LabeledSpan, annotation_field
from dataclasses import dataclass
from pytorch_ie.utils.span import bio_tags_to_spans


@dataclass
class CoNLL2003Document(Document):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")   


class Conll2003(tests.data.test_new_document_and_datasets.GeneratorBasedBuilder):
    DOCUMENT_TYPE = CoNLL2003Document

    BASE_PATH = "conll2003"

    def _generate_document_kwargs(self, dataset):
        return {
            "int_to_str": dataset.features["ner_tags"].feature.int2str
        }

    def _generate_document(self, example, int_to_str):
        doc_id = example["id"]
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        start = 0
        token_offsets = []
        tag_sequence = []
        for token, tag_id in zip(tokens, ner_tags):
            end = start + len(token)
            token_offsets.append((start, end))
            tag_sequence.append(int_to_str(tag_id))

            start = end + 1

        text = " ".join(tokens)
        spans = bio_tags_to_spans(tag_sequence)

        document = CoNLL2003Document(text=text, id=doc_id)

        for label, (start, end) in spans:
            start_offset = token_offsets[start][0]
            end_offset = token_offsets[end][1]
            document.entities.append(
                LabeledSpan(start=start_offset, end=end_offset, label=label)
            )

        return document
