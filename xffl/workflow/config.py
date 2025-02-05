from collections.abc import MutableMapping
from typing import Any


class YamlConfig:
    def __init__(self):
        self.content: MutableMapping[str, Any] = type(self).get_default_content()

    @classmethod
    def get_default_content(cls) -> MutableMapping[str, Any]:
        return {}

    def save(self) -> MutableMapping[str, Any]:
        return self.content
