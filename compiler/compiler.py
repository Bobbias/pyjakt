# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

import errno
from pathlib import Path
from pprint import pprint
from typing import List, Dict, NoReturn

from compiler.error import CompilerError, print_error_json, print_error, eprintln
from compiler.lexing.util import NamedTextFile, FileId, panic


class Compiler:
    file_list: List[NamedTextFile] = []
    errors: List[CompilerError] = []
    file_ids: Dict[str, FileId] = {}
    current_file: FileId = None
    current_file_contents: str = ''
    dump_lexer: bool = False
    dump_parser: bool = False
    dump_type_hints: bool = False
    dump_try_hints: bool = False
    ignore_parser_errors: bool = False
    lexer = None
    json_errors: bool = False
    debug_print: bool = False
    include_paths: List[Path]

    def dbg_print(self, *args, **kwargs):
        if self.debug_print:
            pprint(*args, **kwargs)

    def current_file_id(self):
        return self.current_file
    
    def get_file_path(self, file_id: FileId):
        if file_id.id >= len(self.file_list):
            return None
        return self.file_list[file_id.id]

    def get_file_id_or_register(self, filename: str):
        if filename in self.file_ids.keys():
            file_id = self.file_ids[filename]
        else:
            temp = NamedTextFile(filename)
            self.file_list.append(temp)
            file_id = FileId(len(self.file_list) - 1)
            self.file_ids[filename] = file_id
        return file_id

    def set_current_file(self, file_id: FileId) -> bool:
        old_file_id = self.current_file
        self.current_file = file_id
        try:
            self.current_file_contents = self.file_list[file_id.id].get_contents()
        except FileNotFoundError:
            eprintln(f"\u001b[31;1mError\u001b[0m Could not access "
                     f"{self.file_list[file_id.id].file_path}: File not found")
            self.current_file = old_file_id
            return False
        except PermissionError:
            eprintln(f"\u001b[31;1mError\u001b[0m Could not access "
                     f"{self.file_list[file_id.id].file_path}: Permission denied")
            self.current_file = old_file_id
            return False
        except IOError as e:
            if e.errno == errno.EFBIG:
                eprintln(f"\u001b[31;1mError\u001b[0m Could not access "
                         f"{self.file_list[file_id.id].file_path}: File too big")
            elif e.errno == errno.ENAMETOOLONG:
                eprintln(f"\u001b[31;1mError\u001b[0m Could not access "
                         f"{self.file_list[file_id.id].file_path}: Name too long")
            else:
                panic("Incurred unrecognized error while trying to open file")
            self.current_file = old_file_id
            return False
        return True

    def print_errors(self) -> None:
        idx: int = 0
        for file in self.file_list:
            file_name = file.file_path
            
            for error in self.errors:
                span = error.span

                if span.file_id == idx:
                    if self.json_errors:
                        print_error_json(file_name, error)
                    else:
                        print_error(file_name, file.text, error)

    def search_for_path(self, module_name: str) -> Path | None:
        for include_path in self.include_paths:
            candidate_path = include_path / (module_name + '.jakt')
            if candidate_path.exists():
                return candidate_path

    def panic(self, message: str) -> NoReturn:
        self.print_errors()
        panic(message)
