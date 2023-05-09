from imod.mf6.statusinfo import StatusInfo


class TestSimulation:
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

    def test_iadd_when_adding_two_objects_then_second_object_is_added_to_first(self):
        # Arrange.
        status_info1 = StatusInfo()
        status_info2 = StatusInfo()

        status_info1.add_error("error message 1")
        status_info2.add_error("error message 2")

        # Act.
        status_info1 += status_info2

        # Assert.
        assert len(status_info1.errors) == 2
        assert len(status_info2.errors) == 1
