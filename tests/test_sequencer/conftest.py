"""Every test gets its own user folder (machine lock, shared state), so a
test never touches the real one or waits on another test's lock."""

import pytest


@pytest.fixture(autouse=True)
def _own_user_home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("QDO_SEQUENCER_HOME", str(tmp_path_factory.mktemp("home")))
