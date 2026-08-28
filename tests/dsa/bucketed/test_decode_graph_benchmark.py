import argparse

import pytest

from tests.dsa.bench_qwen3_dsa_decode_graph import parse_batch_sizes


def test_parse_batch_sizes() -> None:
    assert parse_batch_sizes("3,9,33") == [3, 9, 33]


@pytest.mark.parametrize("value", ["", "1", "3,3", "3,nope"])
def test_parse_batch_sizes_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_batch_sizes(value)
