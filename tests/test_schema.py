from typing import Any, Dict, cast

import pystac
import pytest
from pystac.validation.stac_validator import STACValidator

from stac_model.base import JSON
from stac_model.schema import SCHEMA_URI


@pytest.mark.parametrize(
    "mlm_example",  # value passed to 'mlm_example' fixture
    [
        "item_basic.json",
        "item_raster_bands.json",
        "item_eo_bands.json",
        "item_multi_io.json",
    ],
    indirect=True,
)
def test_mlm_schema(
    mlm_validator: STACValidator,
    mlm_example: JSON,
) -> None:
    mlm_item = pystac.Item.from_dict(cast(Dict[str, Any], mlm_example))
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert len(validated) >= len(mlm_item.stac_extensions)  # extra STAC core schemas
    assert SCHEMA_URI in validated


def test_model_metadata_to_dict(eurosat_resnet):
    assert eurosat_resnet.item.to_dict()


def test_validate_model_metadata(eurosat_resnet):
    assert pystac.read_dict(eurosat_resnet.item.to_dict())


def test_validate_model_against_schema(eurosat_resnet, mlm_validator):
    mlm_item = pystac.read_dict(eurosat_resnet.item.to_dict())
    validated = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert SCHEMA_URI in validated
