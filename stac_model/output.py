from enum import Enum
from typing import List, Optional, Union, Any, Dict
from pystac.extensions.classification import Classification
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    PlainSerializer,
)
from typing_extensions import Annotated


class TaskEnum(str, Enum):
    regression = "regression"
    classification = "classification"
    object_detection = "object detection"
    semantic_segmentation = "semantic segmentation"
    instance_segmentation = "instance segmentation"
    panoptic_segmentation = "panoptic segmentation"
    multi_modal = "multi-modal"
    similarity_search = "similarity search"
    image_captioning = "image captioning"
    generative = "generative"
    super_resolution = "super resolution"


class ResultArray(BaseModel):
    shape: List[Union[int, float]]
    dim_names: List[str]
    data_type: str = Field(
        ...,
        pattern="^(uint8|uint16|uint32|uint64|int8|int16|int32|int64|float16|float32|float64)$",
    )

MLMClassification = Annotated[Classification,
                             PlainSerializer(lambda x: x.to_dict(), return_type=Dict[str, Any], when_used='json')]

class ModelOutput(BaseModel):
    task: TaskEnum
    result_array: Optional[List[ResultArray]] = None
    classification_classes: Optional[List[MLMClassification]] = None
    post_processing_function: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
