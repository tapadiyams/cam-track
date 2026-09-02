# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Message-broker abstraction: one interface, two interchangeable backends.

See docs/decisions/0003-message-queue-choice.md for why Redis Streams is
the default and Kafka is kept as a supported alternate backend rather than
the default.
"""

from src.streaming.base import Message, MessageBroker
from src.streaming.factory import get_broker

__all__ = ["Message", "MessageBroker", "get_broker"]
