"""Pruebas de integración de las páginas Streamlit con ``st.testing.AppTest``.

Verifican que cada página se ejecuta sin excepciones con datos reales.
Se usa el modelo ya entrenado del proyecto (``models/linear_regression.joblib``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
PAGINAS = sorted((BASE_DIR / "pages").glob("*.py"))


def _run(page_path: Path):
    app = AppTest.from_file(str(page_path), default_timeout=90)
    app.run()
    return app


@pytest.mark.parametrize("page_path", PAGINAS, ids=[p.stem for p in PAGINAS])
def test_page_runs_without_exception(page_path: Path):
    app = _run(page_path)
    assert not app.exception, f"{page_path.name} lanzó: {[str(e.value) for e in app.exception]}"


@pytest.mark.parametrize("page_path", PAGINAS, ids=[p.stem for p in PAGINAS])
def test_page_renders_content(page_path: Path):
    app = _run(page_path)
    tiene_contenido = bool(app.markdown) or bool(app.dataframe) or bool(app.button) or bool(app.metric)
    assert tiene_contenido, f"{page_path.name} no renderizó contenido"
