"""Unit tests for the `api.run_api` entry point script."""

import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from api import run_api


@pytest.fixture
def fake_uvicorn(monkeypatch):
    """Patch the uvicorn module so no real server spins up."""

    call_args: Dict[str, Any] = {}

    def fake_run(*args, **kwargs):
        call_args["args"] = args
        call_args["kwargs"] = kwargs

    module = SimpleNamespace(run=fake_run)
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return call_args


def _invoke_main(monkeypatch, argv, fake_uvicorn):
    monkeypatch.setattr(sys, "argv", argv)
    run_api.main()
    return fake_uvicorn


def test_main_script_default_development(monkeypatch, capsys, fake_uvicorn):
    uvicorn_call = _invoke_main(monkeypatch, ["python", "api/run_api.py"], fake_uvicorn)

    output = capsys.readouterr().out
    kwargs = uvicorn_call["kwargs"]

    assert "DEVELOPMENT" in output
    assert kwargs["reload"] is True
    assert kwargs["access_log"] is False
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000


@pytest.mark.parametrize("flag", ["--production", "--prod"])
def test_main_script_production_flags(monkeypatch, capsys, fake_uvicorn, flag):
    argv = ["python", "api/run_api.py", flag]
    uvicorn_call = _invoke_main(monkeypatch, argv, fake_uvicorn)

    output = capsys.readouterr().out
    kwargs = uvicorn_call["kwargs"]

    assert "PRODUCTION" in output
    assert kwargs["reload"] is False
    assert kwargs["workers"] == 1


def test_entrypoint_default_production(monkeypatch, capsys, fake_uvicorn):
    uvicorn_call = _invoke_main(monkeypatch, ["alchemist-web"], fake_uvicorn)

    output = capsys.readouterr().out
    kwargs = uvicorn_call["kwargs"]

    assert "PRODUCTION" in output
    assert kwargs["reload"] is False
    assert kwargs["workers"] == 1


@pytest.mark.parametrize("flag", ["--dev", "--development"])
def test_entrypoint_dev_flags(monkeypatch, capsys, fake_uvicorn, flag):
    argv = ["alchemist-web", flag]
    uvicorn_call = _invoke_main(monkeypatch, argv, fake_uvicorn)

    output = capsys.readouterr().out
    kwargs = uvicorn_call["kwargs"]

    assert "DEVELOPMENT" in output
    assert kwargs["reload"] is True
    assert kwargs["access_log"] is False