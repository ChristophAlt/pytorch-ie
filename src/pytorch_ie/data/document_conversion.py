import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar

from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=TokenBasedDocument)


def text_based_document_to_token_based(
    doc: TextBasedDocument,
    tokens: List[str],
    text_span_layers: List[str],
    result_document_type: Type[T],
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    char_to_token: Optional[Callable[[int], Optional[int]]] = None,
    strict: bool = True,
) -> T:
    if char_to_token is None:
        if token_offset_mapping is None:
            raise ValueError(
                "either token_offset_mapping or char_to_token must be provided to convert a text "
                "based document to token based, but both are None"
            )
        char_to_token_dict: Dict[int, int] = {}
        for token_idx, (start, end) in enumerate(token_offset_mapping):
            for char_idx in range(start, end):
                char_to_token_dict[char_idx] = token_idx

        def char_to_token(char_idx: int) -> Optional[int]:
            return char_to_token_dict.get(char_idx)

    result = result_document_type(tokens=tuple(tokens), id=doc.id, metadata=deepcopy(doc.metadata))

    override_annotation_mapping: Dict[str, Dict[int, Annotation]] = {}
    for text_span_layer_name in text_span_layers:
        override_annotation_mapping[text_span_layer_name] = {}
        char_span: Span
        for char_span in doc[text_span_layer_name]:
            start_token_idx = char_to_token(char_span.start)
            end_token_idx_inclusive = char_to_token(char_span.end - 1)
            if start_token_idx is None or end_token_idx_inclusive is None:
                if strict:
                    raise ValueError(
                        f"cannot find token span for char span: {char_span}, "
                        f"token_offset_mapping={token_offset_mapping}"
                    )
                else:
                    logger.warning(f"cannot find token for char span {char_span}, skip it")
                    continue
            token_span = char_span.copy(start=start_token_idx, end=end_token_idx_inclusive + 1)
            override_annotation_mapping[text_span_layer_name][char_span._id] = token_span

        result[text_span_layer_name].extend(
            sorted(
                set(override_annotation_mapping[text_span_layer_name].values()),
                key=lambda span: span.start,
            )
        )

    result.add_all_annotations_from_other(
        doc, override_annotation_mapping=override_annotation_mapping
    )

    # save text, token_offset_mapping and char_to_token (if available) in metadata
    result.metadata["text"] = doc.text
    if token_offset_mapping is not None:
        result.metadata["token_offset_mapping"] = token_offset_mapping
    if char_to_token is not None:
        result.metadata["char_to_token"] = char_to_token

    return result
