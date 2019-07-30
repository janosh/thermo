from utils.decorators import run_once, with_attr


def test_with_attr():
    @with_attr("some_attr", "a_value")
    def some_func():
        pass

    assert some_func.some_attr == "a_value"


def test_run_once():
    @run_once
    def some_func():
        return 42

    assert some_func() == 42
    assert some_func() is None
