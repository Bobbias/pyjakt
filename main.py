# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:
#
#         * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#         * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#         * Neither the name of Bobbias nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

import os
import sys
import logging
from typing import Union
import pathlib

# First party imports
from compiler.compiler import Compiler
from compiler.lexer import Lexer
from compiler.lexing.token import print_token
from compiler.parsing import Parser

this_module = sys.modules[__name__]

##########
# Call main if run as script


def main():
    compiler = Compiler()
    current_directory = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))

    test_file_id = compiler.get_file_id_or_register(r'tests\test1.new')
    file_is_set = compiler.set_current_file(test_file_id)

    if not file_is_set:
        sys.exit(1)

    tokens = Lexer(compiler).lex()

    for token in tokens:
        print_token(token)

    parsed_namespace = Parser.parse(compiler, tokens)


if __name__ == '__main__':
    main()