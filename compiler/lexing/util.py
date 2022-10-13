# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Enables self-referencing in type annotations
from __future__ import annotations

# Third party imports
import re
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import NoReturn

from sumtype import sumtype

# first party imports



def panic(message: str) -> NoReturn:
    from compiler.error import eprintln
    eprintln(f'internal error: {message}')
    sys.exit(1)


def method_logger(method):
    def inner(self, *args, **kwargs):
        ret = method(self, *args, **kwargs)
        logger = logging.getLogger()
        logger.info(f'Call method {method.__name__} of {self} with {args, kwargs} returns {ret}')
        return ret
    return inner


class DebugInfo:
    known_callsites = {}

    def get_color(self, line_no):
        if color := self.known_callsites.get(line_no):
            return color
        else:
            while True:
                value = randint(16, 231)
                if value in self.known_callsites.values():
                    continue
                else:
                    break
            self.known_callsites[line_no] = value
            return value


class LiteralSuffix(sumtype):
    def NONE(): ...
    def UZ(): ...
    def U8(): ...
    def U16(): ...
    def U32(): ...
    def U64(): ...
    def I8(): ...
    def I16(): ...
    def I32(): ...
    def I64(): ...
    def F32(): ...
    def F64(): ...


class NumericConstant(sumtype):
    def I8(value: str): ...
    def I16(value: str): ...
    def I32(value: str): ...
    def I64(value: str): ...
    def U8(value: str): ...
    def U16(value: str): ...
    def U32(value: str): ...
    def U64(value: str): ...
    def F32(value: str): ...
    def F64(value: str): ...

    # for numeric values whose type is not yet determined? is this going to be a thing?
    def UNK(value: str): ...


@dataclass
class FileId:
    id: int = field(init=True, default=-1)


def is_hexdigit(character: str):
    return True if re.match(r'[\da-f]', character, re.RegexFlag.I) else False


def is_octdigit(character: str):
    return True if re.match(r'[0-7]', character) else False


def is_bindigit(character: str):
    return True if character in ['0', '1'] else False


def is_digit(character: str):
    return character.isdigit()


@dataclass(frozen=True)
class TextPosition:
    count: int = field(default=-1)
    row: int = field(default=-1)
    col: int = field(default=-1)


@dataclass(frozen=True)
class TextSpan:
    start: TextPosition = field(default=TextPosition())
    end: TextPosition = field(default=TextPosition())

    def contains(self, span: TextSpan):
        return span.start.count >= self.start.count and span.end.count <= self.end.count


@dataclass(frozen=True)
class FileTextSpan:
    file_id: FileId = field(default_factory=FileId)
    span: TextSpan = field(default=TextSpan())

    def contains(self, span: FileTextSpan):
        return self.file_id == span.file_id and self.span.contains(span.span)


@dataclass(order=False)
class NamedTextFile:
    name: str = field(init=False, default="<script>")
    file_path: str = field(init=True, default="")
    text: str = field(init=False, repr=False, hash=False, compare=False, default='')
    file_id: int = field(init=False, default=-1)

    def __post_init__(self):
        filepath = Path(self.file_path)
        if filepath.is_file():
            self.name = filepath.stem

    def get_contents(self):
        if not self.text:
            filepath = Path(self.file_path)
            with filepath.open(mode='r') as file:
                self.text = file.read()
            return self.text
        return self.text




def literal_suffix_from_string(string: str):
    match string:
        case '':
            return LiteralSuffix.NONE()
        case 'UZ':
            return LiteralSuffix.UZ()
        case 'U8':
            return LiteralSuffix.U8()
        case 'U16':
            return LiteralSuffix.U16()
        case 'U32':
            return LiteralSuffix.U32()
        case 'U64':
            return LiteralSuffix.U64()
        case 'I8':
            return LiteralSuffix.I8()
        case 'I16':
            return LiteralSuffix.I16()
        case 'I32':
            return LiteralSuffix.I32()
        case 'I64':
            return LiteralSuffix.I64()
        case 'F32':
            return LiteralSuffix.F32()
        case 'F64':
            return LiteralSuffix.F64()
        case _:
            return LiteralSuffix.NONE()


# def numeric_constant_type_from_string(string: str):
#     match string:
#         case 'i8':
#             return NumericConstant.I8()
#         case 'i16':
#             return NumericConstant.I16()
#         case 'i32':
#             return NumericConstant.I32()
#         case 'i64':
#             return NumericConstant.I64()
#         case 'u8 ':
#             return NumericConstant.U8()
#         case 'u16':
#             return NumericConstant.U16()
#         case 'u32':
#             return NumericConstant.U32()
#         case 'u64':
#             return NumericConstant.U64()
#         case 'f32':
#             return NumericConstant.F32()
#         case 'f64':
#             return NumericConstant.F64()
#         case _:
#             # Default result... should never happen
#             return NumericConstant.U8()
