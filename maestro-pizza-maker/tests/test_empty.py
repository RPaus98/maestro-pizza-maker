"""Empty test, just to raise warning"""
import warnings


def test_empty() -> None:
    warnings.warn(
        "If this is the only test, this projects is missin any meaningful tests"
    )
