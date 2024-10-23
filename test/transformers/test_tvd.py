from test.utils import assert_verbose_allclose, supports_bfloat16

import pytest
import torch
from liger_kernel.transformers.tvd import LigerTVDLoss

_SHAPE_PARAMS = (
    "B, T, V",
    [
        (1, 4096, 32000),
        (32, 4096, 1024),
        (41, 401, 1271),
        pytest.param(
            1,
            4096,
            128256,
            marks=pytest.mark.skipif(
                torch.cuda.get_device_properties(0).total_memory
                < 36 * 1000 * 1000 * 1000,
                reason="This test requires a GPU with at least 36GB of memory",
            ),
        ),
        (3, 423, 32000),
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        pytest.param(
            torch.bfloat16,
            1e-8,
            5e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(), reason="bfloat16 not supported on this GPU"
            ),
        ),
        (torch.float32, 1e-8, 1e-6),
        (torch.float16, 1e-3, 1e-3),
    ],
)

def reference_tvd_loss(pred, target):
    return 0.5 * (pred - target).abs().sum(dim=-1).mean() 

def _test_tvd_correctness_once(
    target_tvd,
    B,
    T,
    V,
    dtype,
    atol,
    rtol,
    is_last_layer=True,
    device="cuda",
):
    torch.manual_seed(0)

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, device=device).softmax(dim=-1)

    output = reference_tvd_loss(x1, target)
    output2 = target_tvd(x2, target)
    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)

    if (
        not is_last_layer
    ):  # if the loss is the last layer, grad_output is 1.0 and mul op is skipped, testing for that reason
        output = output * 2.0
        output2 = output2 * 2.0

    output.backward()
    output2.backward()
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_tvd_correctness(B, T, V, dtype, atol, rtol):
    liger_tvd = LigerTVDLoss()
    _test_tvd_correctness_once(liger_tvd, B, T, V, dtype, atol, rtol)


@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_tvd_correctness_not_last(B, T, V, dtype, atol, rtol):
    liger_tvd = LigerTVDLoss()
    _test_tvd_correctness_once(
        liger_tvd,
        B,
        T,
        V,
        dtype,
        atol,
        rtol,
        is_last_layer=False,
    )
