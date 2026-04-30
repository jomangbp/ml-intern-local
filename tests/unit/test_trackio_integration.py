from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC
from agent.tools.sandbox_tool import SANDBOX_CREATE_TOOL_SPEC
from agent.tools.trackio_seed import ensure_trackio_dashboard


def test_trackio_seed_helper_imports_with_installed_hub_version():
    assert callable(ensure_trackio_dashboard)


def test_hf_jobs_exposes_trackio_dashboard_args():
    props = HF_JOBS_TOOL_SPEC["parameters"]["properties"]
    assert "trackio_space_id" in props
    assert "trackio_project" in props
    assert "TRACKIO_SPACE_ID" in props["trackio_space_id"]["description"]
    assert "TRACKIO_PROJECT" in props["trackio_project"]["description"]


def test_sandbox_create_exposes_trackio_dashboard_args():
    props = SANDBOX_CREATE_TOOL_SPEC["parameters"]["properties"]
    assert "trackio_space_id" in props
    assert "trackio_project" in props
    assert "TRACKIO_SPACE_ID" in props["trackio_space_id"]["description"]
    assert "TRACKIO_PROJECT" in props["trackio_project"]["description"]
