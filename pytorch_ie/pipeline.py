import collections
import logging
import os
import warnings
from collections import UserDict
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union

import torch
from datasets import Dataset
from packaging import version
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_ie.core.pytorch_ie import PyTorchIEModel
from pytorch_ie.data.document import Document
from pytorch_ie.taskmodules.taskmodule import TaskEncoding, TaskModule, TaskOutput

logger = logging.getLogger(__name__)


class Pipeline:
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Some pipeline, like for instance :class:`~transformers.FeatureExtractionPipeline` (:obj:`'feature-extraction'` )
    output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the :obj:`binary_output` constructor argument. If set to :obj:`True`, the output will be stored in the
    pickle format.
    """

    default_input_names = None

    def __init__(
        self,
        model: PyTorchIEModel,
        taskmodule: TaskModule,
        # args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        **kwargs,
    ):
        self.model = model
        self.taskmodule = taskmodule
        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self.binary_output = binary_output

        self.model = self.model.to(self.device)

        self.call_count = 0
        (
            self._preprocess_params,
            self._forward_params,
            self._postprocess_params,
        ) = self._sanitize_parameters(**kwargs)

    def save_pretrained(self, save_directory: str):
        """
        Save the pipeline's model and taskmodule.

        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)

        self.taskmodule.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples::

            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`, the rest is ignored): The tensors to place on :obj:`self.device`.
            Recursive on lists **only**.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device):
        # if isinstance(inputs, ModelOutput):
        #     return ModelOutput(
        #         {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        #     )
        if isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device)
                for name, tensor in inputs.items()
            }
        elif isinstance(inputs, UserDict):
            return UserDict(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        else:
            return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        preprocess_parameters = {}
        forward_parameters = {}
        postprocess_parameters = {}

        field = pipeline_parameters.get("predict_field")
        if field:
            preprocess_parameters["predict_field"] = field

        return preprocess_parameters, forward_parameters, postprocess_parameters

    def preprocess(
        self, documents: List[Document], predict_field: str, **preprocess_parameters: Dict
    ) -> List[TaskEncoding]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionnary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """

        for document in documents:
            document.clear_predictions(predict_field)

        return self.taskmodule.encode(documents, encode_target=False)

    def _forward(
        self, input_tensors: Tuple[Dict[str, Tensor], Any, Any, Any], **forward_parameters: Dict
    ) -> Dict:
        """
        _forward will receive the prepared dictionnary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        inputs = input_tensors[0]
        return self.model.predict(inputs, **forward_parameters)

    def postprocess(
        self,
        model_inputs: List[TaskEncoding],
        model_outputs: List[TaskOutput],
        **postprocess_parameters: Dict,
    ) -> List[Document]:
        """
        Postprocess will receive the model inputs and (unbatched) model outputs and reformat them into
        something more friendly. Generally it will output a list of documents.
        """
        # This creates annotations from the model outputs and attaches them to the correct documents.
        # IMPORTANT: This might not return the documents in the same order as the input documents!
        return self.taskmodule.decode(encodings=model_inputs, decoded_outputs=model_outputs)

    def get_inference_context(self):
        inference_context = (
            torch.inference_mode
            if version.parse(torch.__version__) >= version.parse("1.9.0")
            else torch.no_grad
        )
        return inference_context

    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            inference_context = self.get_inference_context()
            with inference_context():
                model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                model_outputs = self._forward(model_inputs, **forward_params)
                model_outputs = self._ensure_tensor_on_device(
                    model_outputs, device=torch.device("cpu")
                )
        return model_outputs

    def __call__(
        self,
        documents: Union[Document, List[Document]],
        *args,
        batch_size: int = 1,
        num_workers: int = 8,
        inplace: bool = True,
        is_generative: bool = False,
        **kwargs,
    ) -> Union[Document, List[Document]]:
        if args:
            logger.warning(f"Ignoring args : {args}")
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        self.call_count += 1
        if self.call_count > 10 and self.device.type == "cuda":
            warnings.warn(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset",
                UserWarning,
            )

        single_document = False
        if isinstance(documents, Document):
            single_document = True
            documents = [documents]

        # This creates encodings from the documents. It modifies the documents and may produce multiple entries per
        # document.
        # (Calls: self.taskmodule.encode(documents, encode_target=False))
        model_inputs = self.preprocess(documents, **preprocess_params)

        dataloader: DataLoader[TaskEncoding] = DataLoader(
            model_inputs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.taskmodule.collate,
        )

        model_outputs = []
        with torch.no_grad():
            for batch in dataloader:
                output = self.forward(batch, **forward_params)
                processed_output = self.taskmodule.unbatch_output(output)
                model_outputs.extend(processed_output)
        assert len(model_inputs) == len(
            model_outputs
        ), f"length mismatch: len(model_inputs) [{len(model_inputs)}] != len(model_outputs) [{len(model_outputs)}]"

        documents = self.postprocess(
            model_inputs=model_inputs, model_outputs=model_outputs, **postprocess_params
        )
        if single_document:
            return documents[0]
        else:
            return documents
