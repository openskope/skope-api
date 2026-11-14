"""The two build states (AT-014; OBS-012, OBS-013)."""

import pytest

from skope_release.findings import Report
from skope_release.plan import (
    _PLAN_ISSUER,
    FinalObservation,
    Producer,
    StateError,
    ValidatedBuildPlan,
    require_observation,
    require_plan,
)
from skope_release.preflight import DatasetFiles, preflight_dataset


def test_states_are_issued_only_by_their_stage(temporal_dataset):
    report = Report()
    plan = preflight_dataset(DatasetFiles(temporal_dataset()), release_id=None, mirror=None, producer=Producer("p", "0", "abc"), report=report)
    assert plan is not None and plan._issuer is _PLAN_ISSUER
    with pytest.raises(StateError):
        ValidatedBuildPlan(**{**plan.__dict__, "_issuer": object()})
    with pytest.raises(StateError):
        FinalObservation(**{name: None for name in FinalObservation.__dataclass_fields__})
    with pytest.raises(Exception):
        plan.release_id = "changed"  # frozen


def test_stages_accept_only_their_input_state(temporal_dataset):
    report = Report()
    plan = preflight_dataset(DatasetFiles(temporal_dataset()), release_id=None, mirror=None, producer=Producer("p", "0", "abc"), report=report)
    assert require_plan(plan) is plan
    with pytest.raises(StateError):
        require_plan({"release_id": "x"})
    with pytest.raises(StateError):
        require_observation(plan)  # serializers never consume the plan
