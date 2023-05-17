from imod.mf6.statusinfo import StatusInfo, NestedStatusInfo


class TestStatusInfo:
    def test_add_error_when_message_is_added_then_has_error_returns_true(self):
        # Arrange.
        status_info = StatusInfo()

        # Act.
        status_info.add_error("test error")

        # Assert.
        assert status_info.has_errors()
        assert len(status_info.errors) == 1

    def test_add_error_when_multiple_messages_are_added_then_error_property_contains_multiple_messages(
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


class TestNestedStatusInfo:
    def test_errors_returns_all_nested_errors(self):
        # Arrange.
        root = NestedStatusInfo()
        child1 = NestedStatusInfo()
        child2 = StatusInfo()
        grand_child1 = NestedStatusInfo()
        grand_child2 = StatusInfo()

        root.add(child1)
        root.add(child2)

        child1.add(grand_child1)
        child1.add(grand_child2)

        child2.add_error("test error1")
        grand_child2.add_error("test error2")

        # Act.
        has_errors = root.has_errors()
        errors = root.errors

        # Assert.
        assert has_errors
        assert len(errors) == 2
