from textwrap import dedent

from pytest import fixture

from imod.mf6.statusinfo import NestedStatusInfo, StatusInfo


class TestStatusInfo:
    def test_errors_add_message_has_error(self):
        # Arrange.
        status_info = StatusInfo()

        # Act.
        status_info.add_error("test error")

        # Assert.
        assert status_info.has_errors()
        assert len(status_info.errors) == 1

    def test_errors_add_three_messages_has_three_messages(
        self,
    ):
        # Arrange.
        status_info = StatusInfo()

        # Act.
        status_info.add_error("test error 1")
        status_info.add_error("test error 2")
        status_info.add_error("test error 3")

        # Assert.
        assert status_info.has_errors()
        assert len(status_info.errors) == 3

@fixture(scope="function")
def nested_status_info():
    root = NestedStatusInfo("root")
    child1 = NestedStatusInfo("child1")
    child2 = StatusInfo("child2")
    grand_child1 = NestedStatusInfo("grandchild1")
    grand_child2 = StatusInfo("grandchild2")

    root.add(child1)
    root.add(child2)

    child1.add(grand_child1)
    child1.add(grand_child2)

    child2.add_error("test error1")
    grand_child2.add_error("test error2")
    grand_child2.add_error("test error3")
    return root



class TestNestedStatusInfo:
    def test_flatten_nested_errors(self, nested_status_info):
        # Act.
        has_errors = nested_status_info.has_errors()
        errors = nested_status_info.errors
        # Assert.
        assert has_errors
        assert len(errors) == 2

    def test_set_footer(self, nested_status_info: NestedStatusInfo):
        # Act
        nested_status_info.set_footer_text("footer")
        
        # Assert
        nested_status_info._NestedStatusInfo__footer_text == "footer"

# 'root:\n\t* child1:\n\t\t* grandchild1:\n\t\t\n\t\t\n\t\t* grandchild2:\n\t\t\t* test error2\n\t\n\t\n\t* child2:\n\t\t* test error1\n\n'

    def test_to_string(self, nested_status_info):
        # Arrange
        expected_text = dedent("""root:
        \t* child1
        \t\t* grandchild1:
        \t\t
        \t\t
        \t\t* grandchild2:
        \t\t\t* test error2
        \t\t\t* test error3
        \t
        \t
        \t* child2
        \t\t* test error1


        """)
        # Act
        actual = nested_status_info.to_string()
        # Assert
        assert expected_text == actual

    def test_to_string_with_footer(self, nested_status_info):
        # Arrange
        expected_text = dedent("""\
        \t* test error1
        \t\t* test error2
        footer
        """)
        # Act
        nested_status_info.set_footer_text("footer")
        actual = nested_status_info.to_string()
        # Assert
        assert expected_text == actual