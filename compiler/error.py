# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Enables self-referencing in type annotations
from __future__ import annotations

import sys
from typing import List, Tuple

from sumtype import sumtype

from compiler.lexing.util import TextSpan


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, end=None, **kwargs)


def eprintln(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class CompilerError(sumtype):
    def Message(message: str, span: TextSpan): ...
    def MessageWithHint(message: str, span: TextSpan, hint: str, hint_span: TextSpan): ...


class MessageSeverity(sumtype):
    def Hint(): ...# -> MessageSeverity
    def Error(): ...# -> MessageSeverity

    def color_code(self):
        if self.variant == 'Hint':
            return 94  # Bright Blue
        if self.variant == 'Error':
            return 31  # Red


def print_error_json(file_name: str, error: CompilerError):
    if error.variant == 'Message':
        display_message_with_span_json(MessageSeverity.Error(), file_name, error.message, error.span)
    elif error.variant == 'MessageWithHint':
        display_message_with_span_json(MessageSeverity.Error(), file_name, error.message, error.span)
        display_message_with_span_json(MessageSeverity.Hint(), file_name, error.hint, error.hint_span)


def print_error(file_name: str, file_contents: str, error: CompilerError):
    if error.variant == 'Message':
        display_message_with_span(MessageSeverity.Error(), file_name, file_contents, error.message, error.span)
    elif error.variant == 'MessageWithHint':
        display_message_with_span(MessageSeverity.Error(), file_name, file_contents, error.message, error.span)
        display_message_with_span(MessageSeverity.Hint(), file_name, file_contents, error.hint, error.hint_span)


def display_message_with_span_json(severity: MessageSeverity, file_name: str, message: str, span: TextSpan):
    print(f"{{\"type\":\"diagnostic\",\"message\":\"{message}\","
            f"\"severity\":\"{severity.variant}\","
            f"\"file_id\":{span.file_id},"
            f"\"span\":{{\"start\":{span.start},\"end\":{span.end}}}}}")


def display_message_with_span(severity: MessageSeverity, file_name: str, contents: str | None,
                              message: str, span: TextSpan):
    eprintln(f'{severity.variant}: {message}')

    if not contents:
        return

    line_spans = gather_line_spans(contents)
    line_index = 1
    largest_line_number = len(line_spans)
    width = len(str(largest_line_number))

    while line_index < largest_line_number:
        if line_spans[line_index][0] <= span.start <= line_spans[line_index][1]:
            column_index = span.start - line_spans[line_index][0]
            eprintln(f'----- \x1b[33m{file_name}:{line_index + 1}:{column_index + 1}\x1b0m')
            if line_index > 0:
                print_source_line(severity, contents, line_spans[line_index - 1], span, line_index, largest_line_number)
            print_source_line(severity, contents, line_spans[line_index], span, line_index, largest_line_number)

            for _ in range(span.start - line_spans[line_index][0] + width + 4):
                eprint(' ')

            eprintln(f'\x1b[{severity.color_code()}m^- {message}\x1b[0m')

            while line_index < len(line_spans) and span.end > line_spans[line_index][0]:
                line_index += 1
                if line_index >= len(line_spans):
                    break
                print_source_line(severity, contents, line_spans[line_index], span, line_index + 1, largest_line_number)
                break
        else:
            line_index += 1


def print_source_line(severity: MessageSeverity, file_contents: str,
                      file_span: Tuple[int, int], error_span: TextSpan,
                      line_number: int, largest_line_number: int):
    index = file_span[0]

    eprint(f' {line_number} | ')

    while index <= file_span[1]:
        char = ' '
        if index < file_span[1]:
            char = file_contents[index]
        elif error_span.start == error_span.end and index == error_span.start:
            char = '_'

        if index == error_span.start:
            eprint(f'\x1b[{severity.color_code()}m')
        if index == error_span.end:
            eprint('\x1b[0m')
        eprint(char)

        index += 1
    eprintln('')


def gather_line_spans(file_contents: str) -> [(int, int)]:
    idx = 0
    output: List[Tuple[int, int]] = []

    start = idx
    while idx < len(file_contents):
        if file_contents[idx] == '\n':
            output.append((start, idx))
            start = idx + 1
        idx += 1
    if start < idx:
        output.append((start, idx))
    return output