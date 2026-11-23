"""The demo renders end to end from the committed files in app/data and reports."""

import pytest

from vigil import config

appdata = config.root / "app" / "data"
pytestmark = pytest.mark.skipif(
    not (appdata / "scores.parquet").exists(), reason="run make demo-data first"
)


def test_demo_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.root / "src" / "vigil" / "demo.py"), default_timeout=180)
    app.run()
    assert not app.exception, app.exception
    assert len(app.tabs) == 5
    text = " ".join(m.value for m in app.markdown)
    assert "Approve, review or decline?" in text
    assert "Saved vs rules engine" in text


def test_demo_reacts_to_costs():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.root / "src" / "vigil" / "demo.py"), default_timeout=180)
    app.run()
    before = [m.value for m in app.markdown if "Total cost" in m.value][0]
    app.select_slider[0].set_value(400).run()
    assert not app.exception, app.exception
    after = [m.value for m in app.markdown if "Total cost" in m.value][0]
    assert before != after
