import json
import os
import re
from typing import *

import toml
import yaml
from pydantic import BaseModel

line_break_re = re.compile(r"\r\n|\r|\n")
T = TypeVar("T", bound="BaseConfig")


def parser(fn: Callable):
    def wrapper(cls, file_path_or_str: str):
        if not line_break_re.search(file_path_or_str) and os.path.exists(
            file_path_or_str
        ):
            with open(file_path_or_str) as f:
                file_path_or_str = f.read()
        return fn(cls, file_path_or_str)

    return wrapper


class BaseConfig(BaseModel):
    @classmethod
    def parse_auto(cls: Type[T], filepath: str) -> T:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        if filepath.endswith(".json"):
            return cls.parse_json(filepath)
        elif filepath.endswith(".toml"):
            return cls.parse_toml(filepath)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            return cls.parse_yaml(filepath)
        else:
            raise ValueError(f"Unknown file type: {filepath}")

    @classmethod
    @parser
    def parse_json(cls: Type[T], raw: str) -> T:
        return cls.parse_obj(json.loads(raw))

    @classmethod
    @parser
    def parse_toml(cls: Type[T], raw: str) -> T:
        return cls.parse_obj(toml.loads(raw))

    @classmethod
    @parser
    def parse_yaml(cls: Type[T], raw: str) -> T:
        return cls.parse_obj(yaml.load(raw, Loader=yaml.FullLoader))

    def toml(self):
        return toml.dumps(self.dict())

    def yaml(self):
        return yaml.dump(self.dict())

    def write_json(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.json(indent=4))

    def write_toml(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.toml())

    def write_yaml(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.yaml())
