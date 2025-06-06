from catley import colors
from catley.util.message_log import MessageLog


def test_message_log_stacks_and_revisions() -> None:
    log = MessageLog()
    log.add_message("Hello", colors.WHITE)
    assert len(log.messages) == 1
    assert log.messages[0].full_text == "Hello"
    first_rev = log.revision

    log.add_message("Hello", colors.WHITE)
    assert len(log.messages) == 1
    assert log.messages[0].full_text == "Hello (x2)"
    assert log.revision == first_rev + 1

    log.add_message("Hello", colors.WHITE, stack=False)
    assert len(log.messages) == 2
    assert log.messages[1].full_text == "Hello"
    assert log.revision == first_rev + 2
