import pytest


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'faults_replay_file(path): replay faults from exported decision log',
    )


@pytest.fixture(autouse=True)
def _faults_replay_marker(request, monkeypatch):
    marker = request.node.get_closest_marker('faults_replay_file')
    if marker is not None:
        monkeypatch.setattr(
            request.config.option,
            'faults_replay',
            marker.args[0],
        )
