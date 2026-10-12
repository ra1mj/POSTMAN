import pytest

torch = pytest.importorskip("torch")

from postman.wbc.action_filter import BatchedActionFilter  # noqa: E402


def make_filter(num_envs=3, **kwargs):
    return BatchedActionFilter(
        num_envs,
        lower=torch.full((21,), -2.0),
        upper=torch.full((21,), 4.0),
        velocity_limits=torch.full((21,), 100.0),
        **kwargs,
    )


def measured(num_envs=3, position=1.0):
    state = torch.zeros(num_envs, 25)
    state[:, 4:] = position
    return state


def test_mixed_unit_mapping_and_saturation():
    filt = make_filter(alpha=1, max_wheel_acceleration=1000)
    state = measured()
    filt.reset(torch.arange(3), state)
    raw = torch.zeros(3, 25)
    raw[0, :4] = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    raw[0, 4:] = torch.tensor([0.0] * 20 + [1.0])
    raw[1] = 2.0
    out = filt.step(raw, state)
    torch.testing.assert_close(out[0, :4], torch.tensor([-5.0, -2.5, 2.5, 5.0]))
    torch.testing.assert_close(out[0, 4:-1], torch.ones(20))
    assert out[0, -1] == 2.0  # 1 rad/s limit from initial 1 rad over one step
    assert torch.all(out[1, :4] == 5.0)
    assert torch.all(out[1, 4:] == 2.0)
    assert filt.diagnostics["policy_saturated"][1]


def test_first_order_step_is_monotonic_without_overshoot():
    filt = BatchedActionFilter(
        1, torch.full((21,), -2.0), torch.full((21,), 4.0),
        torch.full((21,), 1000.0), max_wheel_acceleration=1000.0, alpha=0.25,
    )
    state = measured(1)
    filt.reset([0], state)
    action = torch.ones(1, 25)
    values = torch.stack([filt.step(action, state)[0] for _ in range(20)])
    assert torch.all(values[1:] >= values[:-1])
    assert torch.all(values[:, :4] <= 5.0)
    assert torch.all(values[:, 4:] <= 4.0)
    assert torch.all(values[-1] > values[0])


def test_delays_and_partial_reset_are_isolated():
    filt = make_filter(num_envs=2, alpha=1.0, max_wheel_acceleration=1000.0)
    state = measured(2)
    filt.reset([0, 1], state, delays=[0, 2])
    first = torch.ones(2, 25)
    out1 = filt.step(first, state)
    assert out1[0, 0] == 5.0
    assert out1[1, 0] == 0.0
    second = -torch.ones(2, 25)
    out2 = filt.step(second, state)
    assert out2[0, 0] == -5.0
    assert out2[1, 0] == 0.0
    filt.reset([0], measured(1, position=3.0), delays=1)
    out3 = filt.step(torch.zeros(2, 25), state)
    assert out3[0, 0] == 0.0
    assert out3[1, 0] == 5.0  # env 1's first queued command survived env 0 reset
    assert torch.all(out3[0, 4:] == 3.0)


def test_rate_limits_positions_and_wheel_acceleration():
    filt = make_filter(num_envs=1, alpha=1.0)
    state = measured(1)
    filt.reset([0], state)
    out = filt.step(torch.ones(1, 25), state)
    assert torch.all(out[:, :4] == 0.1)
    assert torch.all(out[:, 4:] == 2.0)  # 100 rad/s * 0.01s = 1 rad
    assert filt.diagnostics["rate_limited"][0]


def test_invalid_action_uses_teacher_immediately_and_unsafe_teacher_holds():
    filt = make_filter(num_envs=2, alpha=1.0, max_wheel_acceleration=1000.0)
    state = measured(2)
    filt.reset([0, 1], state, delays=[2, 2])
    filt.step(torch.ones(2, 25), state)
    invalid = torch.zeros(2, 25)
    invalid[0, 0] = float("nan")
    invalid[1, 4] = 1e7
    teacher = measured(2, position=2.0)
    teacher[0, :4] = 1.0
    teacher[1, 4] = float("inf")
    out = filt.step(invalid, state, teacher)
    assert torch.all(torch.isfinite(out))
    assert out[0, 0] == 1.0  # bypass the two-cycle stale queue
    assert torch.all(out[0, 4:] == 2.0)
    assert torch.all(out[1, :4] == 0.0)
    assert torch.all(out[1, 4:] == 1.0)
    assert torch.all(filt.fallback_mask)
    assert filt.diagnostics["teacher_used"][0]
    assert filt.diagnostics["hold_used"][1]
    # A previously queued unsafe command must not emerge after fallback.
    subsequent = filt.step(torch.zeros(2, 25), state)
    assert torch.all(torch.isfinite(subsequent))


def test_bounds_input_and_reset_validation():
    filt = make_filter(num_envs=1)
    state = measured(1)
    with pytest.raises(ValueError):
        filt.reset([0], state, delays=3)
    with pytest.raises(ValueError):
        filt.step(torch.zeros(1, 24), state)
    with pytest.raises(ValueError):
        filt.reset([0], torch.full((1, 25), float("nan")))
