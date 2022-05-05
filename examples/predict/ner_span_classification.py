from dataclasses import dataclass

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument
from pytorch_ie.models import TransformerSpanClassificationModel
from pytorch_ie.pipeline import Pipeline
from pytorch_ie.taskmodules import TransformerSpanClassificationTaskModule


@dataclass
class ExampleDocument(TextDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")


def main():
    model_name_or_path = "pie/example-ner-spanclf-conll03"
    ner_taskmodule = TransformerSpanClassificationTaskModule.from_pretrained(model_name_or_path)
    ner_model = TransformerSpanClassificationModel.from_pretrained(model_name_or_path)

    ner_pipeline = Pipeline(model=ner_model, taskmodule=ner_taskmodule, device=-1)

    document = ExampleDocument(
        "“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at SOSV and managing director of IndieBio."
    )

    ner_pipeline(document, predict_field="entities")

    for entity in document.entities.predictions:
        print(f"{entity} -> {entity.label}")


if __name__ == "__main__":
    main()
