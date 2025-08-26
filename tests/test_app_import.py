def test_import():
    import importlib

    m = importlib.import_module("app.app")
    assert m is not None
