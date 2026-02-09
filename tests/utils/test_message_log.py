from brileta import colors, config
from brileta.util.message_log import MessageLog


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


def test_message_sequence_numbers(monkeypatch) -> None:
    monkeypatch.setattr(config, "SHOW_MESSAGE_SEQUENCE_NUMBERS", True)
    log = MessageLog()
    log.add_message("Hi", colors.WHITE)
    assert log.messages[0].full_text == "[1] Hi"

    log.add_message("Hi", colors.WHITE)
    assert log.messages[0].full_text == "[1] Hi (x2)"

    log.add_message("Bye", colors.WHITE, stack=False)
    assert log.messages[1].full_text == "[3] Bye"
