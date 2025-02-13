import logging
import os
import warnings
from collections import UserDict
from contextlib import contextmanager
from typing import Any, Dict, List, MutableSequence, Optional, Sequence, Tuple, Union

import torch
import tqdm
from packaging import version
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.utils import ModelOutput

from pytorch_ie.core.document import Document
from pytorch_ie.core.model import PyTorchIEModel
from pytorch_ie.core.taskmodule import (
    InplaceNotSupportedException,
    TaskEncoding,
    TaskEncodingDataset,
    TaskModule,
    TaskOutput,
)

logger = logging.getLogger(__name__)


# TODO: use torch.get_autocast_dtype when available
def get_autocast_dtype(device_type: str):
    if device_type == "cuda":
        return torch.float16
    elif device_type == "cpu":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported device type for half precision autocast: {device_type}")


class Pipeline:
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument (see below).

    Args:
        model (:class:`~pytorch_ie.PyTorchIEModel`):
            The deep learning model to use for the pipeline.
        taskmodule (:class:`~pytorch_ie.TaskModule`): The taskmodule to use for encoding
            and decoding the documents.
        device (:obj:`Union[int, str]`, `optional`, defaults to :obj:`"cpu"`):
            The device to run the pipeline on. This can be a CPU device (:obj:`"cpu"`), a GPU
            device (:obj:`"cuda"`) or a specific GPU device (:obj:`"cuda:X"`, where :obj:`X`
            is the index of the GPU).
        half_precision_model (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use half precision model. This can be set to :obj:`True` to reduce
            the memory usage of the model. If set to :obj:`True`, the model will be cast to
            :obj:`torch.float16` on supported devices.
    """

    default_input_names = None

    def __init__(
        self,
        model: PyTorchIEModel,
        taskmodule: TaskModule,
        device: Union[int, str] = "cpu",
        half_precision_model: bool = False,
        compile_model: bool = False,
        **kwargs,
    ):
        self.taskmodule = taskmodule
        device_str = (
            ("cpu" if device < 0 else f"cuda:{device}") if isinstance(device, int) else device
        )
        self.device = torch.device(device_str)

        # Module.to() returns just self, but moved to the device. This is not correctly
        # reflected in typing of PyTorch.
        self.model: PyTorchIEModel = model.to(self.device)  # type: ignore
        if half_precision_model:
            self.model = self.model.to(dtype=get_autocast_dtype(self.device.type))

        if compile_model:
            self.model = torch.compile(self.model)

        self.call_count = 0
        (
            self._preprocess_params,
            self._dataloader_params,
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
        if isinstance(inputs, ModelOutput):
            return type(inputs)(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, dict):
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
            return tuple(self._ensure_tensor_on_device(item, device) for item in inputs)
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        else:
            return inputs

    def _sanitize_parameters(
        self, **pipeline_parameters
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 4 dictionaries of the resolved parameters used by the various `preprocess`,
        `get_dataloader`, `forward` and `postprocess` methods. Do not fill dictionaries if the caller didn't specify
        a kwargs. This lets you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        preprocess_parameters = {}
        dataloader_params = {}
        forward_parameters = {}
        postprocess_parameters: Dict[str, Any] = {}

        # set preprocess parameters
        for p_name in ["document_batch_size"]:
            if p_name in pipeline_parameters:
                preprocess_parameters[p_name] = pipeline_parameters[p_name]

        # set forward parameters
        for p_name in ["show_progress_bar", "fast_dev_run", "half_precision_ops"]:
            if p_name in pipeline_parameters:
                forward_parameters[p_name] = pipeline_parameters[p_name]

        # set dataloader parameters
        for p_name in ["batch_size", "num_workers"]:
            if p_name in pipeline_parameters:
                dataloader_params[p_name] = pipeline_parameters[p_name]

        # set postprocess parameters
        for p_name in ["inplace"]:
            if p_name in pipeline_parameters:
                postprocess_parameters[p_name] = pipeline_parameters[p_name]

        return preprocess_parameters, dataloader_params, forward_parameters, postprocess_parameters

    def preprocess(
        self,
        documents: Sequence[Document],
        document_batch_size: Optional[int] = None,
        **preprocess_parameters: Dict,
    ) -> Sequence[TaskEncoding]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """

        encodings = self.taskmodule.encode(
            documents,
            encode_target=False,
            as_task_encoding_sequence=True,
            document_batch_size=document_batch_size,
        )
        if not isinstance(encodings, Sequence):
            raise TypeError("preprocess has to return a sequence")
        return encodings

    def _forward(
        self, input_tensors: Tuple[Dict[str, Tensor], Any, Any, Any], **forward_parameters: Dict
    ) -> Dict:
        """
        _forward will receive the prepared dictionary from `preprocess` and run it on the model. This method might
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
        model_inputs: Sequence[TaskEncoding],
        model_outputs: Sequence[TaskOutput],
        **postprocess_parameters,
    ) -> Sequence[Document]:
        """
        Postprocess will receive the model inputs and (unbatched) model outputs and reformat them into
        something more friendly. Generally it will output a list of documents.
        """
        # This creates annotations from the model outputs and attaches them to the correct documents.
        return self.taskmodule.decode(
            task_encodings=model_inputs,
            task_outputs=model_outputs,
            **postprocess_parameters,
        )

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

    def get_dataloader(
        self,
        model_inputs: Sequence[TaskEncoding],
        batch_size: int = 1,
        num_workers: int = 8,
        **kwargs,
    ):
        dataloader: DataLoader[TaskEncoding] = DataLoader(
            TaskEncodingDataset(model_inputs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.taskmodule.collate,
            **kwargs,
        )

        return dataloader

    def __call__(
        self,
        documents: Union[Document, Sequence[Document]],
        *args,
        **kwargs,
    ) -> Union[Document, Sequence[Document]]:
        """
        The __call__ method is the entry point for the pipeline. It will run the pipeline workflow in the following
        order:

                1. Encode the documents
                2. Run the model forward pass(es) on the encodings
                3. Combine the model outputs with the input encodings and integrate them back into the documents

        Args:
            documents (:obj:`Union[Document, Sequence[Document]]`): The documents to process. If a single document is
                passed, the output will be a single document. If a list of documents is passed, the output will be a
                list of documents.
            document_batch_size (:obj:`int`, `optional`): The batch size to use for encoding the documents with the
                taskmodule. If not provided, the default batch size of the taskmodule will be used.
            show_progress_bar (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether or not to show a progress bar
                during inference.
            fast_dev_run (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether or not to run a fast development
                run. If set to :obj:`True`, only the first two model inputs will be processed.
            half_precision_ops (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether or not to use half
                precision operations. If set to :obj:`True`, the model will be run with half precision operations
                via :obj:`torch.autocast`.
            batch_size (:obj:`int`, `optional`, defaults to :obj:`1`): The batch size to use for the dataloader. If not
                provided, a batch size of 1 will be used.
            num_workers (:obj:`int`, `optional`, defaults to :obj:`8`): The number of workers to use for the dataloader.
                If not provided, 8 workers will be used.
            inplace (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether or not to modify the input documents
                in place. Requires the input to be a mutable sequence of documents or a single document.

        Note that all the arguments except `documents` can be set in the `__init__` method and/or overridden in the
        `__call__` method.

        Returns:
            :obj:`Union[Document, Sequence[Document]]`: The processed documents. If a single document was passed, a
            single document will be returned. If a list of documents was passed, a list of documents will be returned.
        """
        if args:
            logger.warning(f"Ignoring args : {args}")
        (
            preprocess_params,
            dataloader_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)

        in_place: bool = postprocess_params.get("inplace", True)
        if in_place and not isinstance(documents, (MutableSequence, Document)):
            raise InplaceNotSupportedException(
                "Immutable sequences of Documents (such as Datasets) can't be modified in place. Please set inplace=False."
            )

        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info(
                "Disabling tokenizer parallelism, we're using DataLoader multithreading already"
            )
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        dataloader_params = {**self._dataloader_params, **dataloader_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        single_document = False
        if isinstance(documents, Document):
            single_document = True
            documents = [documents]

        # This creates encodings from the documents. It modifies the documents and may produce multiple entries per
        # document.
        model_inputs = self.preprocess(documents, **preprocess_params)
        if forward_params.pop("fast_dev_run", False):
            warnings.warn(
                "Execute a fast dev run, only the first two model inputs will be processed."
            )
            model_inputs = model_inputs[:2]
        # Create a dataloader from the model inputs. This uses taskmodule.collate().
        dataloader = self.get_dataloader(model_inputs=model_inputs, **dataloader_params)

        show_progress_bar = forward_params.pop("show_progress_bar", False)
        half_precision_ops = forward_params.pop("half_precision_ops", False)
        model_outputs: List = []
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=half_precision_ops):
                for batch in tqdm.tqdm(
                    dataloader, desc="inference", disable=not show_progress_bar
                ):
                    output = self.forward(batch, **forward_params)
                    processed_output = self.taskmodule.unbatch_output(output)
                    model_outputs.extend(processed_output)

        assert len(model_inputs) == len(
            model_outputs
        ), f"length mismatch: len(model_inputs) [{len(model_inputs)}] != len(model_outputs) [{len(model_outputs)}]"

        documents = self.postprocess(
            model_inputs=model_inputs,
            model_outputs=model_outputs,
            **postprocess_params,
        )
        if single_document:
            return documents[0]
        else:
            return documents
