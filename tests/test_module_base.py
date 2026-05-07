import pytest

from models.module import ShapeModule


def test_shape_module_has_no_parameters_and_requires_forward_override() -> None:
    module = ShapeModule()
    assert module.parameters() == ()
    with pytest.raises(NotImplementedError):
        module()
