# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Third party imports
from inspect import currentframe
from re import match, UNICODE
from typing import Callable, Literal, Type, Tuple, Optional

# First party imports
from compiler.lexing.token import Token
from compiler.lexing.util import TextPosition, TextSpan, LiteralSuffix, is_octdigit, is_bindigit, is_digit, \
    is_hexdigit, NumericConstant, DebugInfo


def verify_type(value: int | float, val_type: Type[int | float]):
    # print(f'verifying type for value `{value}`, {type(value)} == {val_type}: {type(value) is val_type}')
    return True if type(value) is val_type else False


def token_from_keyword_or_identifier(string: str, span: TextSpan)->Token:
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
    elif string == 'return':
        return Token.RETURN(span=span)
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
    characters: str
    error_callback: Optional[Callable[[str,TextSpan],None]]

    # internal context
    current_row: int = 0
    current_column: int = 0
    lexing_string: bool = False
    token_start: int = 0
    token_end: int = 0
    index: int = 0
    debug_info = DebugInfo()
    whitespace_chars = [' ', '\t', '\r', '\f', '\v']
    _row: int = 0
    _col: int = 0

    def __init__(self, characters: str, error_callback: Optional[Callable[[str,TextSpan],None]] = None):
        self.characters = characters
        self.error_callback = error_callback

    def __iter__(self):
        return self

    def __next__(self):
        if result := self.next_token():
            if result.variant == 'EOF':
                raise StopIteration
            return result
        else:
            raise StopIteration

    def dbg_print_current(self):
        default_text_color = '\x1b[38;5;250m'
        dark_red = '\x1b[38;5;52m'
        context, current_offset = self.get_context()
        message = (
            "==>" +
            default_text_color + context[:current_offset].encode('unicode_escape').decode('utf-8') +
            dark_red + context[current_offset].encode('unicode_escape').decode('utf-8') +
            default_text_color + context[current_offset+1:].encode('unicode_escape').decode('utf-8')
        )
        print(message)

    def get_context(self)->Tuple[str,int]:
        if self.index > 0:
            start = max(0, self.characters.rfind("\n",0,self.index) + 1)
        else:
            start = 0
        end = self.characters.find("\n",self.index+1)
        if end == -1:
            end = len(self.characters)
        return self.characters[start:end], self.index - start

    def index_inc(self, steps: int = 1, debug: bool = False):
        new_index = min(self.index + steps, len(self.characters))
        for c in self.characters[self.index:new_index]:
            if c == "\n":
                self._col = 0
                self._row += 1
            else:
                self._col += 1
        self.index = new_index
        if debug:
            from inspect import stack
            caller = stack()[1].function
            line = stack()[1].lineno
            color = self.debug_info.get_color(line)
            print(f'\x1b[38;5;{color}m{caller}, {line}\x1b[38;5;250m: self.index += {steps}: {self.index}')

    def error(self, message: str, span: TextSpan = TextSpan()):
        if self.error_callback:
            self.error_callback(message, span)

    def take_while_with_span(self, predicate: Callable[[str], bool])->Tuple[str, TextSpan]:
        n = 0
        while not self.eof() and predicate(self.peek(n)):
            n += 1
        return self.take_with_span(n)

    def take_while(self, predicate: Callable[[str], bool])->str:
        string, _ = self.take_while_with_span(predicate)
        return string

    def take(self, n: int = 1)->str:
        start = self.index
        self.index_inc(n)
        result = self.characters[start:self.index]
        return result

    def take_with_span(self, n: int = 1)->Tuple[str, TextSpan]:
        start = self.position()
        value = self.take(n)
        span = self.span_from(start)
        return value, span

    def position(self)->TextPosition:
        return TextPosition(self.index, self._row, self._col)

    def peek(self, steps: int = 0)->str:
        end = self.index + steps
        if end >= len(self.characters):
            result = ''
        else:
            result = self.characters[self.index + steps]
        return result

    def previous(self)->str:
        index = self.index - 1
        if index < 0:
            return ''
        return self.characters[index]

    def lex_plus(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.PLUS_EQUAL(self.span_from(start))
            case '+':
                self.index_inc()
                return Token.PLUS_PLUS(self.span_from(start))
            case _:
                return Token.PLUS(self.span_from(start))

    def lex_minus(self):
        start = self.position()
        match self.peek():
            case '=':
                self.index_inc(2)
                return Token.MINUS_EQUAL(self.span_from(start))
            case '-':
                self.index_inc(2)
                return Token.MINUS_MINUS(self.span_from(start))
            case '>':
                self.index_inc(2)
                return Token.ARROW(self.span_from(start))
            case _:
                self.index_inc()
                return Token.MINUS(self.span_from(start))

    def lex_asterisk(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.ASTERISKEQUAL(self.span_from(start))
            case _:
                return Token.ASTERISK(self.span_from(start))

    def lex_forward_slash(self)->Token:
        start = self.position()
        self.take()
        next_char = self.peek()
        if next_char == '=':
            self.take()
            return Token.FORWARD_SLASH_EQUAL(self.span_from(start))
        if next_char != '/':
            return Token.FORWARD_SLASH(self.span_from(start))
        # We're in a comment, swallow to end of line.
        self.take_while(lambda c: c != '\n')
        if result := next(self):
            return result
        else:
            return Token.EOF(self.span_from(self.position()))

    def lex_question_mark(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '?':
                self.index_inc()
                if self.peek() == '=':
                    self.index_inc()
                    return Token.QUESTION_MARK_QUESTION_MARK_EQUAL(self.span_from(start))
                return Token.QUESTION_MARK_QUESTION_MARK(self.span_from(start))
            case _:
                return Token.QUESTION_MARK(self.span_from(start))

    def lex_quoted_string(self, delimiter: str)->Token:
        start = self.position()
        self.take()
        if self.eof():
            # error
            return Token.INVALID(self.span(start, start))
        escaped = False
        n = 0
        while not self.eof() and (escaped or self.peek(n) != delimiter):
            if self.peek(n) in ['\r', '\n']:
                pass
            elif not escaped and self.peek(n) == '\\':
                escaped = True
            else:
                escaped = False
            n += 1
        string = self.take(n)
        if self.peek() != delimiter:
            return Token.INVALID(self.span_from(start))
        self.take()
        span = self.span_from(start)
        if delimiter == '\'':
            return Token.SINGLE_QUOTE_STRING(string, span)
        else:
            return Token.QUOTED_STRING(string, span)

    def lex_pipe(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.PIPE_EQUAL(self.span_from(start))
            case _:
                return Token.PIPE(self.span_from(start))

    def lex_ampersand(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.AMPERSAND_EQUAL(self.span_from(start))
            case _:
                return Token.AMPERSAND(self.span_from(start))

    def lex_caret(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.CARET_EQUAL(self.span_from(start))
            case _:
                return Token.CARET(self.span_from(start))

    def lex_percent_sign(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.PERCENT_SIGN_EQUAL(self.span_from(start))
            case _:
                return Token.PERCENT_SIGN(self.span_from(start))

    def lex_exclamation_point(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.NOT_EQUAL(self.span_from(start))
            case _:
                return Token.EXCLAMATION_POINT(self.span_from(start))

    def lex_number_or_name(self)->Token:
        if self.eof():
            self.error('unexpected eof')
            return Token.INVALID(self.span_from(self.position()))
        elif self.peek().isdigit():
            return self.lex_number()
        elif self.peek().isalpha() or self.peek() == '_':
            return self.lex_name()
        else:
            value, span = self.take_with_span(1)
            self.error(f'unexpected character: {value}', span)  # todo: handle span in error stuff
            return Token.INVALID(span)

    def lex_name(self)->Token:
        chars, span = self.take_while_with_span(lambda c:c.isalnum() or c == "_")
        return token_from_keyword_or_identifier(chars, span)

    def consume_numeric_literal_suffix(self, default:LiteralSuffix=LiteralSuffix.I64())->LiteralSuffix:
        next_char = self.peek()
        if next_char not in ['u', 'i', 'f']:
            return default
        elif next_char == 'u' and self.peek(2) == 'z':
            self.index_inc(2)
            return LiteralSuffix.UZ()

        local_index = 1
        width = 0

        # print(f'beginning to consume literal suffix starting at: {self.peek(local_index)}')

        while self.peek(local_index).isdigit():
            if local_index > 3:
                return default
            value = self.peek(local_index)
            local_index += 1
            digit = int(value)
            width = width * 10 + digit

        suffix = default
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

    def lex_number(self)->Token:
        if self.peek() == '0':
            match self.peek(1):
                case 'x':
                    return self.lex_hex_number()
                case 'o':
                    return self.lex_octal_number()
                case 'b':
                    return self.lex_binary_number()
        return self.lex_decimal_number()


    def lex_decimal_number(self)->Token:
        start = self.position()
        value = self.take_while(str.isdigit)
        default_suffix = LiteralSuffix.I64()
        if self.peek() == ".":
            self.take()
            default_suffix = LiteralSuffix.F64()
            value += "." + self.take_while(str.isdigit)
        suffix = self.consume_numeric_literal_suffix(default_suffix)
        span = self.span_from(start)
        return self.make_numeric_constant(value, suffix, span)

    def eof(self):
        return self.index >= len(self.characters)

    def lex_colon(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case ':':
                self.index_inc()
                return Token.COLON_COLON(self.span_from(start))
            case _:
                return Token.COLON(self.span_from(start))

    def lex_less_than(self)->Token:
        start = self.position()
        self.index_inc()

        match self.peek():
            case '=':
                self.index_inc()
                return Token.LESS_THAN_OR_EQUAL(self.span_from(start))
            case '<':
                self.index_inc()
                match self.peek():
                    case '<':
                        self.index_inc()
                        return Token.LEFT_ARITHMETIC_SHIFT(self.span_from(start))
                    case '=':
                        self.index_inc()
                        return Token.LEFT_SHIFT_EQUAL(self.span_from(start))
                    case _:
                        return Token.LEFT_SHIFT(self.span_from(start))
            case _:
                return Token.LESS_THAN(self.span_from(start))

    def lex_greater_than(self):
        start = self.position()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.GREATER_THAN_OR_EQUAL(self.span_from(start))
            case '>':
                self.index_inc()
                match self.peek():
                    case '>':
                        self.index_inc()
                        return Token.RIGHT_ARITHMETIC_SHIFT(self.span_from(start))
                    case '=':
                        self.index_inc()
                        return Token.RIGHT_SHIFT_EQUAL(self.span_from(start))
                    case _:
                        self.index_inc()
                        return Token.RIGHT_SHIFT(self.span_from(start))
            case _:
                self.index_inc()
                return Token.GREATER_THAN(self.span_from(start))

    def lex_dot(self)->Token:
        start = self.position()
        self.index_inc()
        return Token.DOT(self.span_from(start))

    def lex_equals(self)->Token:
        start = self.position()
        self.index_inc()
        match self.peek():
            case '=':
                self.index_inc()
                return Token.DOUBLE_EQUAL(self.span_from(start))
            case '>':
                self.index_inc()
                return Token.FAT_ARROW(self.span_from(start))
            case _:
                return Token.EQUAL(self.span_from(start))

    def span(self, start: TextPosition, end: TextPosition) -> TextSpan:
        return TextSpan(start=start, end=end)

    def span_from(self, start: TextPosition) -> TextSpan:
        return self.span(start, self.position())

    def consume_prefixed_number(self, base=10)->Tuple[str, TextSpan]:
        self.take(2)
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
            case _:
                raise Exception("unknown base %s" % base)
        return self.take_while_with_span(lambda c: testfn(c) or c == "_")

    def make_numeric_constant(self, number: str, suffix: LiteralSuffix, span: TextSpan):
        match suffix.variant:
            case 'U8':
                return Token.NUMBER(NumericConstant.U8(number), span)
            case 'U16':
                return Token.NUMBER(NumericConstant.U16(number), span)
            case 'U32':
                return Token.NUMBER(NumericConstant.U32(number), span)
            case 'U64':
                return Token.NUMBER(NumericConstant.U64(number), span)
            case 'I8':
                return Token.NUMBER(NumericConstant.I8(number), span)
            case 'I16':
                return Token.NUMBER(NumericConstant.I16(number), span)
            case 'I32':
                return Token.NUMBER(NumericConstant.I32(number), span)
            case 'I64':
                return Token.NUMBER(NumericConstant.I64(number), span)
            case 'F32':
                return Token.NUMBER(NumericConstant.F32(number), span)
            case 'U64':
                return Token.NUMBER(NumericConstant.F64(number), span)
            case _:
                return Token.INVALID(span)

    def next_token(self)->Token:
        # Adapted from the jakt compiler's Lexer.next() function.
        # Consume whitespace until a character is encountered or Eof is
        # reached. For Eof return a token.
        self.skip_whitespace()
        if self.eof():
            return Token.EOF(self.span_from(self.position()))

        start = self.position()
        self.dbg_print_current()

        match self.peek():
            case '(':
                self.index_inc()
                return Token.LPAREN(self.span_from(start))
            case ')':
                self.index_inc()
                return Token.RPAREN(self.span_from(start))
            case '[':
                self.index_inc()
                return Token.LSQUARE(self.span_from(start))
            case ']':
                self.index_inc()
                return Token.RSQUARE(self.span_from(start))
            case '{':
                self.index_inc()
                return Token.LCURLY(self.span_from(start))
            case '}':
                self.index_inc()
                return Token.RCURLY(self.span_from(start))
            case '<':
                return self.lex_less_than()
            case '>':
                return self.lex_greater_than()
            case '.':
                return self.lex_dot()
            case ',':
                self.index_inc()
                return Token.COMMA(self.span_from(start))
            case '~':
                self.index_inc()
                return Token.TILDE(self.span_from(start))
            case ';':
                self.index_inc()
                return Token.SEMICOLON(self.span_from(start))
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
                return Token.DOLLAR_SIGN(self.span_from(start))
            case '=':
                return self.lex_equals()
            case '\n':
                self.index_inc()
                return Token.EOL(self.span_from(start))
            case '\'':
                return self.lex_quoted_string(delimiter='\'')
            case '\"':
                return self.lex_quoted_string(delimiter='\"')
            case 'b':
                return self.lex_character_constant_or_name()
            case 'c':
                return self.lex_character_constant_or_name()
            case '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '0':
                return self.lex_number()
            case _:
                return self.lex_number_or_name()

    def lex_character_constant_or_name(self):
        if self.peek() != '\'':
            return self.lex_number_or_name()

        is_byte = self.peek() == 'b'
        if is_byte:
            self.index_inc()

        start = self.position()
        self.index_inc()
        first = self.peek()
        second = self.peek(1)
        escaped = False

        while not self.eof() and (escaped or self.peek() != '\''):
            if (escaped and (self.index - start > 3)) or self.index - start > 2:
                break

            if not escaped and self.peek() == '\\':
                escaped = True

            self.index_inc()

        if self.eof() or self.peek() != '\'':
            self.error('Expected single quote', self.span(start, start))
        self.index_inc()

        quote = first
        if escaped:
            quote += second

        if is_byte:
            return Token.SINGLE_QUOTE_BYTE_STRING(quote, self.span_from(start))
        else:
            return Token.SINGLE_QUOTE_STRING(quote, self.span_from(start))

    def skip_whitespace(self):
        self.take_while(lambda c: c in self.whitespace_chars)

    def lex_hex_number(self)->Token:
        value, span = self.consume_prefixed_number(base=16)
        if value.endswith('_'):
            self.error('Hexadecimal number literal cannot end with underscore', span)
            return Token.INVALID(span)
        suffix = self.consume_numeric_literal_suffix()
        return self.make_numeric_constant(value, suffix, span)

    def lex_octal_number(self)->Token:
        value, span = self.consume_prefixed_number(base=8)
        if self.previous() == '_':
            self.error('Octal number literal cannot end with underscore', span)
            return Token.INVALID(span)
        suffix = self.consume_numeric_literal_suffix()
        if value.endswith('_'):
            self.error('Could not parse octal number', span)
            return Token.INVALID(span)
        return self.make_numeric_constant(value, suffix, span)

    def lex_binary_number(self)->Token:
        value, span = self.consume_prefixed_number(base=2)
        if value.endswith('_'):
            self.error('Binary number literal cannot end with underscore', span)
            return Token.INVALID(span)
        suffix = self.consume_numeric_literal_suffix()
        if self.peek().isalnum():
            self.error('Could not parse binary number', span)
            return Token.INVALID(span)
        return self.make_numeric_constant(value, suffix, span)
