def test_infer_delimwhitespace():
    from imod.formats.gen.gen import infer_delimwhitespace

    assert infer_delimwhitespace("1 2 3", 3) == (True, True)
    assert infer_delimwhitespace("1,2,3", 3) == (False, True)
    assert infer_delimwhitespace("1,2,3", 4) == (False, False)
    assert infer_delimwhitespace("1, 2, 3", 3) == (False, True)
    assert infer_delimwhitespace("1\t2\t3", 3) == (True, True)
    assert infer_delimwhitespace("1 2,3", 3) == (False, False)
