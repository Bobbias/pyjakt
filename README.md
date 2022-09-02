## Unnamed-compiler

This project is a compiler for an as-yet unnamed language based syntactically off of [Jakt](https://github.com/SerenityOS/jakt/).

Much of the current code is based off of translating the Jakt self-hosting source into python. Future plans include native code generation via [LLVM](https://llvm.org).

## Project Status

[x] Lexing
[x] Parsing
[ ] Typechecking
[ ] Codegen

Parsing is mostly complete, however the sumtype dependency is causing some issues which must be fixed before continuing to typechecking.

## Dependencies

### Current

* [sumtype](https://github.com/lubieowoce/sumtype)

### Future

* [LLVM](https://llvm.org/)

## Building

run `python3 main.py`

## License

SPDX-License-Identifier: BSD-3-Clause-Clear

Copyright (c) 2022-2022 Blair 'Bobbias' Stacey

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        * Neither the name of Bobbias nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 