# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Third party imports
from re import match, UNICODE
from typing import List, Type

# First party imports
from compiler.compiler import Compiler
from compiler.error import CompilerError
from compiler.lexing.token import Token
from compiler.lexing.util import NamedTextFile, TextSpan, LiteralSuffix, is_octdigit, is_bindigit, is_digit, \
    is_hexdigit, NumericConstant, DebugInfo, FileId


def verify_type(value: int | float, val_type: Type[int | float]):
    # print(f'verifying type for value `{value}`, {type(value)} == {val_type}: {type(value) is val_type}')
    return True if type(value) is val_type else False


def token_from_keyword_or_identifier(string: str, span: TextSpan):
    if string == 'and':
        return Token.AND(span=span)
    elif string == 'anon':
        return Token.ANON(span=span)
    elif string == 'as':
        return Token.AS(span=span)
    elif string == 'boxed':
        return Token.BOXED(span=span)
    elif string == 'break':
        return Token.BREAK(span=span)
    elif string == 'catch':
        return Token.CATCH(span=span)
    elif string == 'class':
        return Token.CLASS(span=span)
    elif string == 'continue':
        return Token.CONTINUE(span=span)
    elif string == 'comptime':
        return Token.COMPTIME(span=span)
    elif string == 'cpp':
        return Token.CPP(span=span)
    elif string == 'defer':
        return Token.DEFER(span=span)
    elif string == 'else':
        return Token.ELSE(span=span)
    elif string == 'enum':
        return Token.ENUM(span=span)
    elif string == 'extern':
        return Token.EXTERN(span=span)
    elif string == 'false':
        return Token.FALSE(span=span)
    elif string == 'for':
        return Token.FOR(span=span)
    elif string == 'function':
        return Token.FUNCTION(span=span)
    elif string == 'guard':
        return Token.GUARD(span=span)
    elif string == 'if':
        return Token.IF(span=span)
    elif string == 'import':
        return Token.IMPORT(span=span)
    elif string == 'in':
        return Token.IN(span=span)
    elif string == 'is':
        return Token.IS(span=span)
    elif string == 'let':
        return Token.LET(span=span)
    elif string == 'loop':
        return Token.LOOP(span=span)
    elif string == 'match':
        return Token.MATCH(span=span)
    elif string == 'mut':
        return Token.MUT(span=span)
    elif string == 'namespace':
        return Token.NAMESPACE(span=span)
    elif string == 'not':
        return Token.NOT(span=span)
    elif string == 'or':
        return Token.OR(span=span)
    elif string == 'private':
        return Token.PRIVATE(span=span)
    elif string == 'public':
        return Token.PUBLIC(span=span)
    elif string == 'raw':
        return Token.RAW(span=span)
    elif string == 'restricted':
        return Token.RESTRICTED(span=span)
    elif string == 'struct':
        return Token.STRUCT(span=span)
    elif string == 'this':
        return Token.THIS(span=span)
    elif string == 'throw':
        return Token.THROW(span=span)
    elif string == 'throws':
        return Token.THROWS(span=span)
    elif string == 'true':
        return Token.TRUE(span=span)
    elif string == 'try':
        return Token.TRY(span=span)
    elif string == 'unsafe':
        return Token.UNSAFE(span=span)
    elif string == 'weak':
        return Token.WEAK(span=span)
    elif string == 'while':
        return Token.WHILE(span=span)
    elif string == 'yield':
        return Token.YIELD(span=span)
    else:
        return Token.IDENTIFIER(name=string, span=span)


class Lexer:
    file: FileId
    compiler: Compiler

    # internal context
    current_row: int = 0
    current_column: int = 0

    lexing_string: bool = False

    token_start: int = 0
    token_end: int = 0

    _spaces_per_tab: int = 4
    _value_string: str = ''
    _floating: bool = False
    _number_too_large: bool = False

    index: int = 0

    debug_info = DebugInfo()

    whitespace_chars = [' ', '\t', '\r', '\f', '\v']

    def __init__(self, compiler: Compiler):
        self.compiler = compiler
        self.file = compiler.current_file

    def __iter__(self):
        # ensure we have a file and compiler before trying to lex
        if not self.file or not self.compiler:
            raise ValueError('Must provide a file to lex')

        # Note: This may be unnecessary, or it may be unnecessary to provide defaults above
        self.current_row = 0
        self.current_column = 0
        self.lexing_string = False
        self.token_start = 0
        self.token_end = 0
        self._spaces_per_tab = 4
        self.index = 0
        return self

    def __next__(self):
        if result := self.tokenize_file():
            if result.variant == 'EOF':
                raise StopIteration
            return result
        else:
            raise StopIteration

    def index_inc(self, steps: int = 1, debug: bool =False):
        self.index += steps
        if debug:
            from inspect import stack
            caller = stack()[1].function
            line = stack()[1].lineno
            color = self.debug_info.get_color(line)
            print(f'\x1b[38;5;{color}m{caller}, {line}\x1b[38;5;250m: self.index += {steps}: {self.index}')

    def error(self, message: str, span: TextSpan = TextSpan()):
        self.compiler.errors.append(CompilerError.Message(message, span))

    def lex(self):
        tokens: List[Token] = []

        for token in self:
            tokens.append(token)

        return tokens

    def peek(self, steps: int = 1):
        end = self.index + steps
        if end >= len(self.compiler.current_file_contents):
            return ''
        return self.compiler.current_file_contents[self.index + steps]

    def peek_behind(self, steps: int = 1):
        index = self.index - steps
        if index < 0:
            return ''
        return self.compiler.current_file_contents[index]

    def lex_plus(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.PLUS_EQUAL(self.span(start, self.index))
            case '+':
                self.index_inc()
                return Token.PLUS_PLUS(self.span(start, self.index))
            case _:
                return Token.PLUS(self.span(self.index - 1, self.index))

    def lex_minus(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.MINUS_EQUAL(self.span(start, self.index))
            case '-':
                self.index_inc()
                return Token.MINUS_MINUS(self.span(start, self.index))
            case '>':
                self.index_inc()
                return Token.ARROW(self.span(start, self.index))
            case _:
                return Token.MINUS(self.span(self.index - 1, self.index))

    def lex_asterisk(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.ASTERISKEQUAL(self.span(start, self.index))
            case _:
                return Token.ASTERISK(self.span(start, self.index))

    def lex_forward_slash(self):
        start = self.index
        self.index_inc()
        next_char = self.peek()
        if next_char == '=':
            self.index_inc()
            return Token.FORWARD_SLASH_EQUAL(self.span(start, self.index))
        if next_char != '/':
            return Token.FORWARD_SLASH(self.span(start, self.index))
        # We're in a comment, swallow to end of line.
        while not self.eof():
            next_char = self.peek()
            self.index_inc()
            if next_char == '\n':
                break
        if result := self.__next__():
            return result
        else:
            return Token.EOF(self.span(self.index, self.index))

    def lex_question_mark(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '?':
                self.index_inc()
                if self.peek() == '=':
                    self.index_inc()
                    return Token.QUESTION_MARK_QUESTION_MARK_EQUAL(self.span(start, self.index))
                return Token.QUESTION_MARK_QUESTION_MARK(self.span(start, self.index))
            case _:
                return Token.QUESTION_MARK(self.span(start, self.index))

    def lex_quoted_string(self, delimiter: str):
        start = self.index
        self.index_inc()

        if self.eof():
            # error
            return Token.INVALID(self.span(start, start))

        escaped = False

        while not self.eof() and (escaped or self.peek() != delimiter):
            next_char = self.peek()
            if next_char in ['\r', '\n']:
                self.index_inc()
                continue

            if not escaped and self.peek() == '\\':
                escaped = True
            else:
                escaped = False
            self.index_inc()

        self.index_inc()
        end = self.index

        string = self.compiler.current_file_contents[start + 1: end]

        if self.compiler.current_file_contents[self.index] == delimiter:
            # print(f'delimiter reached: {self.compiler.current_file_contents[self.index]}')
            if not self.eof():
                self.index_inc()

        if delimiter == '\'':
            return Token.SINGLE_QUOTE_STRING(string, self.span(start, end))
        return Token.QUOTED_STRING(string, self.span(start, end))

    def lex_pipe(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.PIPE_EQUAL(self.span(start, self.index))
            case _:
                return Token.PIPE(self.span(start, self.index))

    def lex_ampersand(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.AMPERSAND_EQUAL(self.span(start, self.index))
            case _:
                return Token.AMPERSAND(self.span(start, self.index))

    def lex_caret(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.CARET_EQUAL(self.span(start, self.index))
            case _:
                return Token.CARET(self.span(start, self.index))

    def lex_percent_sign(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.PERCENT_SIGN_EQUAL(self.span(start, self.index))
            case _:
                return Token.PERCENT_SIGN(self.span(start, self.index))

    def lex_exclamation_point(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.NOT_EQUAL(self.span(start, self.index))
            case _:
                return Token.EXCLAMATION_POINT(self.span(start, self.index))

    def lex_number_or_name(self):  # -> Token
        start = self.index

        # guard against stray EOF
        if self.eof():
            self.error('unexpected eof')
            return Token.INVALID(self.span(self.index, self.index + 1))

        # Figure out if we're lexing a number
        if self.peek().isdigit():
            return self.lex_number()
        # or a name
        elif self.peek().isalpha() or self.peek() == '_':
            while self.peek().isalnum() or self.peek() == '_':
                self.index_inc()
            self.index_inc() # note: is this correct?
            end = self.index
            chars = self.compiler.current_file_contents[start:end]
            span = self.span(start, end)
            return token_from_keyword_or_identifier(chars, span)

        # due to some inconsistencies in index incrementing, these following if statements detect
        # whether we're looking at a single character, or a single digit.
        if self.compiler.current_file_contents[self.index].isalpha() and not self.peek().isalpha():
            end = self.index + 1
            span = self.span(start, end)
            char = self.compiler.current_file_contents[self.index]
            self.index_inc()
            return token_from_keyword_or_identifier(char, span)

        if self.compiler.current_file_contents[self.index].isdigit() and not self.peek().isdigit():
            val = int(self.compiler.current_file_contents[self.index])
            self.index_inc()
            return Token.NUMBER(NumericConstant.U64(val), self.span(start, self.index))

        self.index_inc()
        unknown_char = self.compiler.current_file_contents[self.index]
        end = self.index
        span = self.span(start, end)
        self.error(f'unexpected character: {unknown_char}', span)  # todo: handle span in error stuff
        return Token.INVALID(span)

    def consume_numeric_literal_suffix(self):
        next_char = self.compiler.current_file_contents[self.index]
        if next_char not in ['u', 'i', 'f']:
            # print('consume_numeric_literal_suffix: next_char is not `u`, `i` or `f`')
            # print(f'next_char: {next_char}')
            return None
        elif next_char == 'u' and self.peek(2) == 'z':
            self.index_inc(2)
            return LiteralSuffix.UZ()

        local_index = 1
        width = 0

        # print(f'beginning to consume literal suffix starting at: {self.peek(local_index)}')

        while self.peek(local_index).isdigit():
            if local_index > 3:
                return None
            value = self.compiler.current_file_contents[self.index + local_index]
            local_index += 1
            digit = int(value)
            width = width * 10 + digit

        suffix = None
        match next_char:
            case 'u':
                match width:
                    case 8:
                        suffix = LiteralSuffix.U8()
                    case 16:
                        suffix = LiteralSuffix.U16()
                    case 32:
                        suffix = LiteralSuffix.U32()
                    case 64:
                        suffix = LiteralSuffix.U64()
            case 'i':
                match width:
                    case 8:
                        suffix = LiteralSuffix.I8()
                    case 16:
                        suffix = LiteralSuffix.I16()
                    case 32:
                        suffix = LiteralSuffix.I32()
                    case 64:
                        suffix = LiteralSuffix.I64()
            case 'f':
                match width:
                    case 32:
                        suffix = LiteralSuffix.F32()
                    case 64:
                        suffix = LiteralSuffix.F64()

        self.index_inc(local_index)
        return suffix

    def lex_number(self):
        start: int = self.index
        number_too_large: bool = False
        floating: bool = False
        self._value_string = ''
        self._floating = False
        self._number_too_large = False

        if self.peek() == '0':
            match self.peek(2):
                case 'x':
                    return self.lex_hex_number(start)
                case 'o':
                    return self.lex_octal_number(start)
                case 'b':
                    return self.lex_binary_number(start)

        self.handle_floats()

        end = self.index
        span = self.span(start, end)

        if number_too_large:
            # err
            return Token.INVALID(span)

        if self.peek() == '_':
            # err
            return Token.INVALID(span)

        default_suffix = LiteralSuffix.NONE()
        if floating:
            default_suffix = LiteralSuffix.F64

        # print('checking next char for literal suffix')
        result = self.consume_numeric_literal_suffix()
        # print(f'result = {result}')
        suffix = result if result else default_suffix
        # print(f'suffix is: {suffix}')

        is_float_suffix = True if suffix in [LiteralSuffix.F64, LiteralSuffix.F32] else False

        if floating and not is_float_suffix:
            return Token.INVALID(span)

        # print(f'value_string = {self._value_string}')

        if floating:
            value = float(self._value_string)
        else:
            value = int(self._value_string)

        token = self.make_numeric_constant(value, suffix, span)
        # print(f'token = {token}')
        return token

    def eof(self):
        return self.index >= len(self.compiler.current_file_contents)

    def lex_colon(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case ':':
                return Token.COLON_COLON(self.span(start, self.index + 1))
            case _:
                return Token.COLON(self.span(start, self.index))

    def lex_less_than(self):
        start = self.index
        self.index_inc()

        match self.compiler.current_file_contents[self.index]:
            case '=':
                self.index_inc()
                return Token.LESS_THAN_OR_EQUAL(self.span(start, self.index))
            case '<':
                self.index_inc()
                match self.peek():
                    case '<':
                        self.index_inc()
                        return Token.LEFT_ARITHMETIC_SHIFT(self.span(start, self.index))
                    case '=':
                        self.index_inc()
                        return Token.LEFT_SHIFT_EQUAL(self.span(start, self.index))
                    case _:
                        return Token.LEFT_SHIFT(self.span(self.index - 1, self.index))
            case _:
                return Token.LESS_THAN(self.span(self.index - 1, self.index))

    def lex_greater_than(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                return Token.GREATER_THAN_OR_EQUAL(self.span(start, self.index))
            case '>':
                self.index_inc()
                match self.peek():
                    case '>':
                        self.index_inc()
                        return Token.RIGHT_ARITHMETIC_SHIFT(self.span(start, self.index))
                    case '=':
                        self.index_inc()
                        return Token.RIGHT_SHIFT_EQUAL(self.span(start, self.index))
                    case _:
                        return Token.RIGHT_SHIFT(self.span(self.index - 1, self.index))
            case _:
                return Token.GREATER_THAN(self.span(self.index - 1, self.index))

    def lex_dot(self):
        start = self.index
        self.index_inc()
        return Token.DOT(self.span(start, self.index))

    def lex_equals(self):
        start = self.index
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.DOUBLE_EQUAL(self.span(start, self.index))
            case '>':
                self.index_inc()
                return Token.FAT_ARROW(self.span(start, self.index))
            case _:
                return Token.EQUAL(self.span(start, self.index))

    def current_indentation(self):
        result = len(match(r"\s*", self.compiler.current_file_contents.expandtabs(4), UNICODE).group(0))
        return result

    def span(self, start: int, end: int) -> TextSpan:
        return TextSpan(file_id=self.compiler.current_file, start=start, end=end)

    def lex_prefixed_number(self, base=10):
        self.index_inc(2)
        start = self.index
        testfn = None
        match base:
            case 2:
                testfn = is_bindigit
            case 8:
                testfn = is_octdigit
            case 10:
                testfn = is_digit
            case 16:
                testfn = is_hexdigit

        while testfn(self.peek()):
            if self.peek() == '_':
                self.index_inc()
        string = self.compiler.current_file_contents[start:self.index]
        return int(string, base)

    def make_numeric_constant(self, number: int | float, suffix: LiteralSuffix, span: TextSpan):
        # print('making numeric constant')
        # print(f'suffix variant = {suffix.variant}')
        match suffix.variant:
            case 'U8':
                return Token.NUMBER(NumericConstant.U8(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'U16':
                return Token.NUMBER(NumericConstant.U16(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'U32':
                return Token.NUMBER(NumericConstant.U32(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'U64':
                return Token.NUMBER(NumericConstant.U64(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'I8':
                return Token.NUMBER(NumericConstant.I8(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'I16':
                return Token.NUMBER(NumericConstant.I16(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'I32':
                return Token.NUMBER(NumericConstant.I32(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'I64':
                return Token.NUMBER(NumericConstant.I64(number), span) \
                    if verify_type(number, int) else Token.INVALID(span)
            case 'F32':
                return Token.NUMBER(NumericConstant.F32(number), span) \
                    if verify_type(number, float) else Token.INVALID(span)
            case 'U64':
                return Token.NUMBER(NumericConstant.F64(number), span) \
                    if verify_type(number, float) else Token.INVALID(span)
            case _:
                return Token.INVALID(span)

    def tokenize_file(self):  # -> Token
        # Adapted from the jakt compiler's Lexer.next() function.
        # Consume whitespace until a character is encountered or Eof is
        # reached. For Eof return a token.
        # print(f'index: {self.index}, file length: {len(self.compiler.current_file_contents)}')
        while True:
            if self.index >= len(self.compiler.current_file_contents):
                return Token.EOF(self.span(self.index - 1, self.index - 1))
            if self.compiler.current_file_contents[self.index] in self.whitespace_chars:
                self.index_inc()
            else:
                break

        # while self.compiler.current_file_contents[self.index] in [' ', '']:
        #     print('ignoring extraneous space or empty string, this should not happen')
        #     self.index_inc()

        start = self.index
        current_character = self.compiler.current_file_contents[self.index].encode('unicode_escape').decode('utf-8')
        # print(f'tokenize_file: current character = `{current_character}`')
        # print(f'tokenize_file: next character = `{self.peek().encode("unicode_escape").decode("utf-8")}`')

        match self.compiler.current_file_contents[self.index]:
            case '(':
                self.index_inc()
                return Token.LPAREN(self.span(start, self.index))
            case ')':
                self.index_inc()
                return Token.RPAREN(self.span(start, self.index))
            case '[':
                self.index_inc()
                return Token.LSQUARE(self.span(start, self.index))
            case ']':
                self.index_inc()
                return Token.RSQUARE(self.span(start, self.index))
            case '{':
                self.index_inc()
                return Token.LCURLY(self.span(start, self.index))
            case '}':
                self.index_inc()
                return Token.RCURLY(self.span(start, self.index))
            case '<':
                return self.lex_less_than()
            case '>':
                return self.lex_greater_than()
            case '.':
                return self.lex_dot()
            case ',':
                self.index_inc()
                return Token.COMMA(self.span(start, self.index))
            case '~':
                self.index_inc()
                return Token.TILDE(self.span(start, self.index))
            case ';':
                self.index_inc()
                return Token.SEMICOLON(self.span(start, self.index))
            case ':':
                return self.lex_colon()
            case '?':
                return self.lex_question_mark()
            case '+':
                return self.lex_plus()
            case '-':
                return self.lex_minus()
            case '*':
                return self.lex_asterisk()
            case '/':
                return self.lex_forward_slash()
            case '^':
                return self.lex_caret()
            case '|':
                return self.lex_pipe()
            case '%':
                return self.lex_percent_sign()
            case '!':
                return self.lex_exclamation_point()
            case '&':
                return self.lex_ampersand()
            case '$':
                self.index_inc()
                return Token.DOLLAR_SIGN(self.span(start, self.index))
            case '=':
                return self.lex_equals()
            case '\n':
                self.index_inc()
                return Token.EOL(self.span(start, self.index))
            case '\'':
                return self.lex_quoted_string(delimiter='\'')
            case '\"':
                return self.lex_quoted_string(delimiter='\"')
            case _:
                return self.lex_number_or_name()

    def lex_hex_number(self, start: int):
        value: int = self.lex_prefixed_number(base=16)
        end = self.index
        span = self.span(start, end)
        if self.peek_behind() == '_':
            self.error('Hexadecimal number literal cannot end with underscore', span)
            return Token.INVALID(span)
        suffix = self.consume_numeric_literal_suffix()
        return self.make_numeric_constant(value, suffix, span)

    def lex_octal_number(self, start: int):
        value: int = self.lex_prefixed_number(base=8)
        end = self.index
        span = self.span(start, end)
        if self.peek_behind() == '_':
            self.error('Octal number literal cannot end with underscore', span)
            return Token.INVALID(span)
        suffix = self.consume_numeric_literal_suffix()
        if self.peek().isalnum():
            self.error('Could not parse octal number', span)
            return Token.INVALID(span)
        return self.make_numeric_constant(value, suffix, span)

    def lex_binary_number(self, start: int):
        value: int = self.lex_prefixed_number(base=2)
        end = self.index
        span = self.span(start, end)
        if self.peek_behind() == '_':
            self.error('Binary number literal cannot end with underscore', span)
            return Token.INVALID(span)
        suffix = self.consume_numeric_literal_suffix()
        if self.peek().isalnum():
            self.error('Could not parse octal number', span)
            return Token.INVALID(span)
        return self.make_numeric_constant(value, suffix, span)

    def handle_floats(self):
        start = self.index
        # print('inside handle_floats')
        while not self.eof():
            digit = self.compiler.current_file_contents[self.index]
            # print(f'digit = {digit}')
            # print(f'start = {start}, index = {self.index+1}')
            if digit == '.':
                # print('found decimal')
                if not self.peek().isdigit() or self._floating:
                    break
                self._floating = True
                self.index_inc()
                continue
            elif not digit.isdigit():
                # print(f'`{digit}` is not a digit, breaking')
                break

            self._value_string = self.compiler.current_file_contents[start:self.index+1]
            # print(f'value_string = {self._value_string}')
            self.index_inc()

            if not self._floating:
                if int(self._value_string).bit_length() > 64:
                    self._number_too_large = True
            else:
                try:
                    float(self._value_string)
                except ValueError:
                    self.error('Cannot convert number to float', self.span(start, self.index))
                    return Token.INVALID(self.span(start, self.index))
                except OverflowError:
                    self.error('Float number too large', self.span(start, self.index))
                    self._number_too_large = True

            if self.peek() == '_':
                if self.peek(2).isdigit():
                    self.index_inc()
                else:
                    break
