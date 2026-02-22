def test_import_culearn():
    import culearn
    assert hasattr(culearn, "__version__")