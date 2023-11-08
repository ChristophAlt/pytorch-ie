from .document import Annotation, AnnotationLayer, Document, annotation_field
from .metric import DocumentMetric
from .model import PyTorchIEModel
from .module_mixins import RequiresDocumentTypeMixin
from .statistic import DocumentStatistic
from .taskmodule import TaskEncoding, TaskModule

# backwards compatibility
AnnotationList = AnnotationLayer
