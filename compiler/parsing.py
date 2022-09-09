# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

# Third party imports
from typing import List, Tuple, Union, Any

from sumtype import sumtype

from compiler.compiler import Compiler
from compiler.error import CompilerError
from compiler.lexing.token import Token
from compiler.lexing.util import TextSpan, NumericConstant, DebugInfo

from compiler.parsedtypes import ParsedField, ValueEnumVariant, SumEnumVariant, EnumVariantPatternArgument, ParsedBlock, \
    ParsedParameter, ParsedVarDecl, ParsedCall, ParsedMatchCase, ParsedFunction, ParsedNamespace, ParsedExternImport, \
    ParsedModuleImport, ImportName, ParsedRecord, ParsedMethod, ParsedGenericParameter, ParsedVariable


class ParsedType(sumtype):
    def Name(name: str, span: TextSpan):
        ...  # -> ParsedType

    def NamespacedName(name: str, namespaces: List[str], params: List[Any], span: TextSpan):
        ...  # -> ParsedType

    def GenericType(name: str, generic_parameters: List[Any], span: TextSpan):
        ...  # -> ParsedType

    def Array(inner: Any, span: TextSpan):
        ...  # -> ParsedType

    def Dictionary(key: Any, value: Any, span: TextSpan):
        ...  # -> ParsedType

    def Tuple(types: List[Any], span: TextSpan):
        ...  # -> ParsedType

    def Set(inner: Any, span: TextSpan):
        ...  # -> ParsedType

    def Optional(inner: Any, span: TextSpan):
        ...  # -> ParsedType

    def Reference(inner: Any, span: TextSpan):
        ...  # -> ParsedType

    def MutableReference(inner: Any, span: TextSpan):
        ...  # -> ParsedType

    def RawPointer(inner: Any, span: TextSpan):
        ...  # -> ParsedType

    def WeakPointer(inner: Any, span: TextSpan):
        ...  # -> ParsedType

    def Function(params: List[ParsedParameter], can_throw: bool, return_type: Any, span: TextSpan):
        ...  # -> ParsedType

    def Empty(span: TextSpan):
        ...  # -> ParsedType

    def __eq__(self, other):
        if self.variant != other.variant:
            return False
        if self.variant == 'Name':
            return self.name == other.name
        elif self.variant in ['Array', 'Set', 'Optional', 'Reference', 'MutableReference', 'RawPointer', 'WeakPointer']:
            return self.inner == other.inner
        elif self.variant == 'NamespacedName':
            if len(self.namespaces) != len(other.namespaces):
                return False
            for lhs_ns, rhs_ns in zip(self.namespaces, other.namespaces):
                if lhs_ns != rhs_ns:
                    return False
            if len(self.params) != len(other.params):
                return False
            for lhs_param, rhs_param in zip(self.params, other.params):
                if lhs_param != rhs_param:
                    return False
            return self.name == other.name
        elif self.variant == 'GenericType':
            if len(self.generic_parameters) != len(other.generic_parameters):
                return False
            for lhs_generic_param, rhs_generic_param in zip(self.generic_parameters, other.generic_parameters):
                if lhs_generic_param != rhs_generic_param:
                    return False
            return self.name == other.name
        elif self.variant == 'Function':
            if len(self.params) != len(other.params):
                return False
            for lhs_param, rhs_param in zip(self.params, other.params):
                if lhs_param != rhs_param:
                    return False
            return self.can_throw == other.can_throw and self.return_type == other.return_type
        elif self.variant == 'Dictionary':
            return self.key == other.key and self.value == other.value
        elif self.variant == 'Tuple':
            if len(self.types) != len(other.types):
                return False
            for lhs_type, rhs_type in zip(self.types, other.types):
                if lhs_type != rhs_type:
                    return False
            return True


class DefinitionLinkage(sumtype):
    def Internal(): ...  # -> DefinitionLinkage

    def External(): ...  # -> DefinitionLinkage


class RecordType(sumtype):
    def Struct(fields: List[ParsedField]): ...  # -> RecordType

    def Class(fields: List[ParsedField], super_class: ParsedType): ...  # -> RecordType

    def ValueEnum(underlying_type: ParsedType, variants: List[ValueEnumVariant]): ...  # -> RecordType

    def SumEnum(is_boxed: bool, variants: List[SumEnumVariant]): ...  # -> RecordType

    def Invalid(): ...  # -> RecordType


class BinaryOperator(sumtype):
    def Add(): ...  # -> BinaryOperator

    def Subtract(): ...  # -> BinaryOperator

    def Multiply(): ...  # -> BinaryOperator

    def Divide(): ...  # -> BinaryOperator

    def Modulo(): ...  # -> BinaryOperator

    def LessThan(): ...  # -> BinaryOperator

    def LessThanOrEqual(): ...  # -> BinaryOperator

    def GreaterThan(): ...  # -> BinaryOperator

    def GreaterThanOrEqual(): ...  # -> BinaryOperator

    def Equal(): ...  # -> BinaryOperator

    def NotEqual(): ...  # -> BinaryOperator

    def BitwiseAnd(): ...  # -> BinaryOperator

    def BitwiseXor(): ...  # -> BinaryOperator

    def BitwiseOr(): ...  # -> BinaryOperator

    def BitwiseLeftShift(): ...  # -> BinaryOperator

    def BitwiseRightShift(): ...  # -> BinaryOperator

    def ArithmeticLeftShift(): ...  # -> BinaryOperator

    def ArithmeticRightShift(): ...  # -> BinaryOperator

    def LogicalAnd(): ...  # -> BinaryOperator

    def LogicalOr(): ...  # -> BinaryOperator

    def NoneCoalescing(): ...  # -> BinaryOperator

    def Assign(): ...  # -> BinaryOperator

    def BitwiseAndAssign(): ...  # -> BinaryOperator

    def BitwiseXorAssign(): ...  # -> BinaryOperator

    def BitwiseOrAssign(): ...  # -> BinaryOperator

    def BitwiseLeftShiftAssign(): ...  # -> BinaryOperator

    def BitwiseRightShiftAssign(): ...  # -> BinaryOperator

    def AddAssign(): ...  # -> BinaryOperator

    def SubtractAssign(): ...  # -> BinaryOperator

    def MultiplyAssign(): ...  # -> BinaryOperator

    def ModuloAssign(): ...  # -> BinaryOperator

    def DivideAssign(): ...  # -> BinaryOperator

    def NoneCoalescingAssign(): ...  # -> BinaryOperator

    def Invalid(): ...  # -> BinaryOperator

    # Note: All assignments must contain Assign in the name for this to work correctly.
    #       But I think this is better than making a manual list of the names.
    def is_assignment(self):
        return 'Assign' in self.variant

    def __eq__(self, other: BinaryOperator):
        return self.variant == other.variant


class TypeCast(sumtype):
    def Fallible(cast: ParsedType): ...  # -> TypeCast

    def Infallible(cast: ParsedType): ...  # -> TypeCast

    def __eq__(self, other: TypeCast):
        if self.variant != other.variant:
            return False
        return self.cast == other.cast


class UnaryOperator(sumtype):
    def PreIncrement():
        ...  # -> UnaryOperator

    def PostIncrement():
        ...  # -> UnaryOperator

    def PreDecrement():
        ...  # -> UnaryOperator

    def PostDecrement():
        ...  # -> UnaryOperator

    def Negate():
        ...  # -> UnaryOperator

    def Dereference():
        ...  # -> UnaryOperator

    def RawAddress():
        ...  # -> UnaryOperator

    def Reference():
        ...  # -> UnaryOperator

    def MutableReference():
        ...  # -> UnaryOperator

    def LogicalNot():
        ...  # -> UnaryOperator

    def BitwiseNot():
        ...  # -> UnaryOperator

    def TypeCast(cast: TypeCast):
        ...  # -> UnaryOperator

    def Is(type_name: ParsedType):
        ...  # -> UnaryOperator

    def IsEnumVariant(inner: ParsedType, bindings: List[EnumVariantPatternArgument]):
        ...  # -> UnaryOperator

    def __eq__(self, other: UnaryOperator):
        if self.variant != other.variant:
            return False
        if self.variant == 'TypeCast':
            return self.cast == other.cast
        elif self.variant == 'Is':
            return self.type_name == other.type_name
        elif self.variant == 'InEnumVariant':
            if len(self.bindings) != len(other.bindings):
                return False
            for lhs_binding, rhs_binding in zip(self.bindings, other.bindings):
                if lhs_binding != rhs_binding:
                    return False
            return self.inner == other.inner


class FunctionType(sumtype):
    def Normal(): ...  # -> FunctionType

    def ImplicitConstructor(): ...  # -> FunctionType

    def ImplicitEnumConstructor(): ...  # -> FunctionType

    def ExternalClassConstructor(): ...  # -> FunctionType


class FunctionLinkage(sumtype):
    def Internal(): ...  # -> FunctionLinkage

    def External(): ...  # -> FunctionLinkage


class ParsedMatchPattern(sumtype):
    def EnumVariant(variant_name: List[Tuple[str, TextSpan]],
                    variant_arguments: List[EnumVariantPatternArgument],
                    arguments_span: TextSpan): ...  # -> ParsedMatchPattern

    def Expression(expr: Any): ...  # -> ParsedMatchPattern

    def CatchAll(): ...  # -> ParsedMatchPattern


class ParsedMatchBody(sumtype):
    def Expression(expr: Any):
        ...  # -> ParsedMatchBody

    def Block(block: ParsedBlock):
        ...  # -> ParsedMatchBody

    def __eq__(self, other: ParsedMatchBody):
        if self.variant != other.variant:
            return False
        if self.variant == 'Expression':
            return self.expr == other.expr
        if self.variant == 'Block':
            return self.block == other.block


class ParsedCapture(sumtype):
    def ByValue(name: str, span: TextSpan): ...  # -> ParsedCapture

    def ByReference(name: str, span: TextSpan): ...  # -> ParsedCapture

    def ByMutableReference(name: str, span: TextSpan): ...  # -> ParsedCapture


class ParsedStatement(sumtype):
    def Expression(expr: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def Defer(statement: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def UnsafeBlock(block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedStatement

    def DestructuringAssignment(vars_: List[ParsedVarDecl], var_decl: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def VarDecl(var: ParsedVarDecl, init: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def If(condition: Any, then_block: ParsedBlock, else_statement: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def Block(block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedStatement

    def Loop(block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedStatement

    def While(condition: Any, block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedStatement

    def For(iterator_name: str, name_span: TextSpan, range_: Any, block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedStatement

    def Break(span: TextSpan):
        ...  # -> ParsedStatement

    def Continue(span: TextSpan):
        ...  # -> ParsedStatement

    def Return(expr: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def Throw(expr: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def Yield(expr: Any, span: TextSpan):
        ...  # -> ParsedStatement

    def InlineCpp(block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedStatement

    def Guard(expr: Any, else_block: ParsedBlock, remaining_code: ParsedBlock, span: TextSpan):
        ...  # -> ParsedStatement

    def Invalid(span: TextSpan):
        ...  # -> ParsedStatement

    def __eq__(self, other: ParsedStatement):
        if self.variant != other.variant:
            return False
        if self.variant in ['Expression', 'Throw', 'Yield']:
            return self.expr == other.expr
        elif self.variant in ['UnsafeBlock', 'Block', 'Loop', 'InlineCpp']:
            return self.block == other.block
        elif self.variant == 'Defer':
            return self.stmt == other.stmt
        elif self.variant == 'DestructuringAssignment':
            if len(self.vars_) != len(other.vars_):
                return False
            for lhs_var, rhs_var in zip(self.vars_, other.vars_):
                if lhs_var != rhs_var:
                    return False
            if self.var_decl != other.var_decl:
                return False
            return True
        elif self.variant == 'If':
            return self.condition == other.condition and self.then_block == other.then_block and self.else_statement == other.else_statement
        elif self.variant == 'VarDecl':
            return self.var == other.var and self.init == other.init
        elif self.variant == 'While':
            return self.condition == other.condition and self.block == other.block
        elif self.variant == 'For':
            return self.iterator_name == other.iterator_name and self.range_ == other.range_ and self.block == other.block
        elif self.variant == 'Return':
            # if both are empty, they're the same
            if not self.expr and not other.expr:
                return True
            else:  # otherwise we check their exprs
                return self.expr == other.expr
        elif self.variant == 'Guard':
            return self.expr == other.expr and self.else_block == other.else_block


class Visibility(sumtype):
    def Public(): ...  # -> Visibility

    def Private(): ...  # -> Visibility

    def Restricted(whitelist: List[ParsedType], span: TextSpan): ...  # -> Visibility


class ParsedExpression(sumtype):
    def Boolean(val: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def NumericConstant(val: NumericConstant, span: TextSpan):
        ...  # -> ParsedExpression

    def QuotedString(val: str, span: TextSpan):
        ...  # -> ParsedExpression

    def SingleQuotedString(val: str, span: TextSpan):
        ...  # -> ParsedExpression

    def SingleQuotedByteString(quote: str, span: TextSpan):
        ...  # -> ParsedExpression

    def Call(call: ParsedCall, span: TextSpan):
        ...  # -> ParsedExpression

    def MethodCall(expr: Any, call: ParsedCall, is_optional: bool, span: TextSpan):
        ...  # -> ParsedExpression

    def IndexedTuple(expr: Any, index: int, is_optional: bool, span: TextSpan):
        ...  # -> ParsedExpression

    def IndexedStruct(expr: Any, field: str, is_optional: bool, span: TextSpan):
        ...  # -> ParsedExpression

    def Var(name: str, span: TextSpan):
        ...  # -> ParsedExpression

    def IndexedExpression(expr: Any, index: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def IndexedRangeExpression(expr: Any, from_: Any,
                               to: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def UnaryOp(expr: Any, op: UnaryOperator, span: TextSpan):
        ...  # -> ParsedExpression

    def BinaryOp(lhs: Any, op: BinaryOperator, rhs: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def Operator(op: BinaryOperator, span: TextSpan):
        ...  # -> ParsedExpression

    def OptionalSome(expr: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def OptionalNone(span: TextSpan):
        ...  # -> ParsedExpression

    def Array(values_: List[Any], fill_size: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def Dictionary(values_: List[Any], span: TextSpan):
        ...  # -> ParsedExpression

    def Set(values_: List[Any], span: TextSpan):
        ...  # -> ParsedExpression

    def Tuple(values_: List[Any], span: TextSpan):
        ...  # -> ParsedExpression

    def Range(from_: Any, to: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def ForcedUnwrap(expr: Any, span: TextSpan):
        ...  # -> ParsedExpression

    def Match(expr: Any, cases: List[ParsedMatchCase], span: TextSpan):
        ...  # -> ParsedExpression

    def EnumVariantArg(expr: Any, arg: EnumVariantPatternArgument,
                       enum_variant: ParsedType, span: TextSpan):
        ...  # -> ParsedExpression

    def NamespacedVar(name: str, namespace: List[str], span: TextSpan):
        ...  # -> ParsedExpression

    def Function(captures: List[ParsedCapture], params: List[ParsedParameter], can_throw: bool,
                 return_type: ParsedType, block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedExpression

    def Try(expr: Any, catch_block: ParsedBlock, catch_name: str, span: TextSpan):
        ...  # -> ParsedExpression

    def TryBlock(stmt: Any, error_name: str,
                 error_span: TextSpan, catch_block: ParsedBlock, span: TextSpan):
        ...  # -> ParsedExpression

    def Invalid(span: TextSpan):
        ...  # -> ParsedExpression

    def precedence(self) -> int:
        if self.variant in ['Multiply', 'Divide', 'Modulo']:
            return 100
        if self.variant in ['Add', 'Subtract']:
            return 90
        if self.variant in ['BitwiseLeftShift', 'BitwiseRightShift', 'ArithmeticLeftShift', 'ArithmeticRightShift']:
            return 85
        if self.variant in ['LessThan', 'LessThanOrEqual', 'GreaterThan', 'GreaterThanOrEqual', 'Equal', 'NotEqual']:
            return 80
        if self.variant == 'BitwiseAnd':
            return 73
        if self.variant == 'BitwiseXor':
            return 72
        if self.variant == 'BitwiseOr':
            return 71
        if self.variant == 'LogicalAnd':
            return 70
        if self.variant in ['LogicalOr', 'NoneCoalescing']:
            return 69
        if self.variant in ['Assign', 'BitwiseAndAssign', 'BitwiseOrAssign', 'BitwiseXorAssign',
                            'BitwiseLeftShiftAssign', 'BitwiseRightShiftAssign', 'AddAssign',
                            'SubtractAssign', 'MultiplyAssign', 'DivideAssign', 'NoneCoalescingAssign']:
            return 50
        return 0


class Parser:
    index: int
    tokens: List[Token]
    compiler: Compiler

    debug_info = DebugInfo()

    def index_inc(self, steps: int = 1, debug: bool = False):
        self.index += steps
        if debug:
            from inspect import stack
            caller = stack()[1].function
            line = stack()[1].lineno
            color = self.debug_info.get_color(line)
            print(f'\x1b[38;5;{color}m{caller}, {line}\x1b[38;5;250m: self.index += {steps}: {self.index}')

    def __init__(self, index: int, compiler: Compiler, tokens: List[Token]):
        self.index = index
        self.compiler = compiler
        self.tokens = tokens

    def span(self, start, end):
        return TextSpan(self.compiler.current_file, start, end)

    def empty_span(self):
        return self.span(0, 0)

    def error(self, message: str, span: TextSpan):
        if not self.compiler.ignore_parser_errors:
            self.compiler.errors.append(CompilerError.Message(message, span))

    def error_with_hint(self, message: str, span: TextSpan, hint: str, hint_span: TextSpan):
        if not self.compiler.ignore_parser_errors:
            self.compiler.errors.append(CompilerError.MessageWithHint(message, span, hint, hint_span))

    def eof(self):
        return self.index >= (len(self.tokens) - 1)

    def eol(self):
        return self.eof() or (self.tokens[self.index].variant == 'EOL')

    def peek(self, steps: int = 1):
        if self.eof() or self.index + steps >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.index + steps]

    def previous(self):
        if self.index == 0 or self.index > len(self.tokens):
            return Token.EOF(self.span(0, 0))
        return self.tokens[self.index - 1]

    def current(self):
        return self.peek(0)

    def skip_newlines(self):
        while self.current().variant == 'EOL':
            self.index_inc()

    def parse_expression(self, allow_assignments: bool, allow_newlines: bool):
        expr_stack = []
        last_precedence = 1000000

        lhs = self.parse_operand()
        expr_stack.append(lhs)

        op: BinaryOperator = BinaryOperator.Invalid()
        while True:
            if allow_newlines:
                if self.eof() or self.current().variant == 'LCURLY':
                    break
            else:
                if self.eol():
                    break
            parsed_operator = self.parse_operator(allow_assignments)
            if parsed_operator.variant == 'Invalid':
                break
            precedence = parsed_operator.precedence()
            self.skip_newlines()
            rhs = self.parse_operand()

            while precedence <= last_precedence and len(expr_stack) > 1:  # was 1
                rhs = expr_stack.pop()
                op = expr_stack.pop()

                if last_precedence < precedence:
                    expr_stack.append(op)
                    expr_stack.append(rhs)
                    break

                lhs = expr_stack.pop()

                match op.variant:
                    case 'Operator':
                        new_span = self.merge_spans(lhs.span, rhs.span)
                        expr_stack.append(ParsedExpression.BinaryOp(lhs, op, rhs, new_span))
                    case _:
                        self.error('Operator is not an operator', op.span)
            expr_stack.append(parsed_operator)
            expr_stack.append(rhs)
            last_precedence = precedence

        while len(expr_stack) > 1:
            rhs = expr_stack.pop()
            parsed_operator = expr_stack.pop()
            lhs = expr_stack.pop()

            match parsed_operator.variant:
                case 'Operator':
                    new_span = self.merge_spans(lhs.span, rhs.span)
                    expr_stack.append(ParsedExpression.BinaryOp(lhs, parsed_operator.op, rhs, new_span))
                case _:
                    self.error('Operator is not an operator', op.span)

        return expr_stack[0]

    def parse_operator(self, allow_assignments: bool):
        span = self.current().span
        op: BinaryOperator | None = None
        token_type = self.current().variant
        if token_type == 'QUESTION_MARK_QUESTION_MARK':
            op = BinaryOperator.NoneCoalescing()
        elif token_type == 'PLUS':
            op = BinaryOperator.Add()
        elif token_type == 'MINUS':
            op = BinaryOperator.Subtract()
        elif token_type == 'ASTERISK':
            op = BinaryOperator.Multiply()
        elif token_type == 'FORWARD_SLASH':
            op = BinaryOperator.Divide()
        elif token_type == 'PERCENT_SIGN':
            op = BinaryOperator.Modulo()
        elif token_type == 'AND':
            op = BinaryOperator.LogicalAnd()
        elif token_type == 'OR':
            op = BinaryOperator.LogicalOr()
        elif token_type == 'DOUBLE_EQUAL':
            op = BinaryOperator.Equal()
        elif token_type == 'NOT_EQUAL':
            op = BinaryOperator.NotEqual()
        elif token_type == 'LESS_THAN':
            op = BinaryOperator.LessThan()
        elif token_type == 'LESS_THAN_OR_EQUAL':
            op = BinaryOperator.LessThanOrEqual()
        elif token_type == 'GREATER_THAN':
            op = BinaryOperator.GreaterThan()
        elif token_type == 'GREATER_THAN_OR_EQUAL':
            op = BinaryOperator.GreaterThanOrEqual()
        elif token_type == 'AMPERSAND':
            op = BinaryOperator.BitwiseAnd()
        elif token_type == 'PIPE':
            op = BinaryOperator.BitwiseOr()
        elif token_type == 'CARET':
            op = BinaryOperator.BitwiseXor()
        elif token_type == 'LEFT_SHIFT':
            op = BinaryOperator.BitwiseLeftShift()
        elif token_type == 'RIGHT_SHIFT':
            op = BinaryOperator.BitwiseRightShift()
        elif token_type == 'LEFT_ARITHMETIC_SHIFT':
            op = BinaryOperator.ArithmeticLeftShift()
        elif token_type == 'RIGHT_ARITHMETIC_SHIFT':
            op = BinaryOperator.ArithmeticRightShift()
        elif token_type == 'EQUAL':
            op = BinaryOperator.Assign()
        elif token_type == 'LEFT_SHIFT_EQUAL':
            op = BinaryOperator.BitwiseLeftShiftAssign()
        elif token_type == 'RIGHT_SHIFT_EQUAL':
            op = BinaryOperator.BitwiseRightShiftAssign()
        elif token_type == 'AMPERSAND_EQUAL':
            op = BinaryOperator.BitwiseAndAssign()
        elif token_type == 'PIPE_EQUAL':
            op = BinaryOperator.BitwiseOrAssign()
        elif token_type == 'CARET_EQUAL':
            op = BinaryOperator.BitwiseXorAssign()
        elif token_type == 'PLUS_EQUAL':
            op = BinaryOperator.AddAssign()
        elif token_type == 'MINUS_EQUAL':
            op = BinaryOperator.SubtractAssign()
        elif token_type == 'ASTERISK_EQUAL':
            op = BinaryOperator.MultiplyAssign()
        elif token_type == 'FORWARD_SLASH_EQUAL':
            op = BinaryOperator.DivideAssign()
        elif token_type == 'PERCENT_SIGN_EQUAL':
            op = BinaryOperator.ModuloAssign()
        elif token_type == 'QUESTION_MARK_QUESTION_MARK_EQUAL':
            op = BinaryOperator.NoneCoalescingAssign()

        if not op:
            return ParsedExpression.Invalid(span)

        self.index_inc()

        if allow_assignments and op.is_assignment():
            self.error('Assignment is not allowed in this position', span)
            return ParsedExpression.Operator(op, span)
        return ParsedExpression.Operator(op, span)

    def parse_operand(self):
        self.skip_newlines()
        start = self.current().span
        self.skip_newlines()
        expr = self.parse_operand_base()
        return self.parse_operand_postfix_operator(start, expr)

    def parse_operand_base(self) -> ParsedExpression:
        if self.current().variant == 'DOT':
            return ParsedExpression.Var('this', self.current().span)
        elif self.current().variant == 'TRY':
            span = self.current().span
            self.index_inc()
            if self.current().variant == 'LCURLY':
                return self.parse_try_block()
            else:
                expression = self.parse_expression(allow_assignments=True, allow_newlines=True)
                catch_block = None
                catch_name = None
                if self.current().variant == 'CATCH':
                    self.index_inc()
                    if self.current().variant == 'IDENTIFIER':
                        catch_name = self.current().name
                        self.index_inc()
                    catch_block = self.parse_block()
                return ParsedExpression.Try(expression, catch_block, catch_name, span)
        elif self.current().variant == 'QUOTED_STRING':
            quote = self.current().string
            span = self.current().span
            self.index_inc()
            return ParsedExpression.QuotedString(quote, span)
        elif self.current().variant == 'SINGLE_QUOTED_STRING':
            quote = self.current().string
            span = self.current().span
            self.index_inc()
            return ParsedExpression.SingleQuotedString(quote, span)
        elif self.current().variant == 'SINGLE_QUOTED_BYTE_STRING':
            quote = self.current().quote
            span = self.current().span
            self.index_inc()
            return ParsedExpression.SingleQuotedByteString(quote, span)
        elif self.current().variant == 'NUMBER':
            val = self.current().value
            span = self.current().span
            self.index_inc()
            return ParsedExpression.NumericConstant(val, span)
        elif self.current().variant == 'TRUE':
            span = self.current().span
            self.index_inc()
            return ParsedExpression.Boolean(True, span)
        elif self.current().variant == 'FALSE':
            span = self.current().span
            self.index_inc()
            return ParsedExpression.Boolean(False, span)
        elif self.current().variant == 'THIS':
            span = self.current().span
            self.index_inc()
            return ParsedExpression.Var('this', span)
        elif self.current().variant == 'NOT':
            start = self.current().span
            self.index_inc()
            expr = self.parse_operand()
            span = self.merge_spans(start, expr.span)
            return ParsedExpression.UnaryOp(expr, UnaryOperator.LogicalNot(), span)
        elif self.current().variant == 'TILDE':
            start = self.current().span
            self.index_inc()
            expr = self.parse_operand()
            span = self.merge_spans(start, expr.span)
            return ParsedExpression.UnaryOp(expr, UnaryOperator.BitwiseNot(), span)
        elif self.current().variant == 'IDENTIFIER':
            name = self.current().name
            span = self.current().span
            if self.peek().variant == 'LPAREN':
                if name == 'Some':
                    self.index_inc()
                    expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                    return ParsedExpression.OptionalSome(expr, span)
                call = self.parse_call()
                return ParsedExpression.Call(call, span)
            if self.peek().variant == 'LESS_THAN':
                self.compiler.ignore_parser_errors = True
                call = self.parse_call()
                self.compiler.ignore_parser_errors = False
                if not call:
                    if name == 'None':
                        return ParsedExpression.OptionalNone(span)
                    else:
                        return ParsedExpression.Var(name, span)
                return ParsedExpression.Call(call, span)
            self.index_inc()
            if name == 'None':
                return ParsedExpression.OptionalNone(span)
            return ParsedExpression.Var(name, span)
        elif self.current().variant == 'LPAREN':
            start_span = self.current().span
            self.index_inc()
            expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
            self.skip_newlines()

            match self.current().variant:
                case 'RPAREN':
                    self.index_inc()
                case 'COMMA':
                    self.index_inc()
                    tuple_exprs = [expr]
                    end_span = start_span

                    while not self.eof():
                        if self.current().variant in ['EOL', 'COMMA']:
                            self.index_inc()
                        elif self.current().variant == 'RPAREN':
                            self.index_inc()
                            break
                        else:
                            expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                            end_span = expr.span
                            tuple_exprs.append(expr)
                    if self.eof():
                        self.error('Expected `)`', self.current().span)

                    expr = ParsedExpression.Tuple(tuple_exprs, self.merge_spans(start_span, end_span))
                case _:
                    self.error('Expected `)`', self.current().span)
            return expr
        elif self.current().variant in ['PLUSPLUS', 'MINUSMINUS', 'MINUS']:
            op: UnaryOperator | None = None
            if self.current().variant == 'PLUS_PLUS':
                op = UnaryOperator.PreIncrement()
            elif self.current().variant == 'MINUS_MINUS':
                op = UnaryOperator.PreDecrement()
            elif self.current().variant == 'MINUS':
                op = UnaryOperator.Negate()
            else:
                self.error('Something went wrong parsing unary operators PreIncrement, PreDecrement and Negate',
                           self.current().span)
            start = self.current().span
            self.index_inc()
            expr = self.parse_operand()
            span = self.merge_spans(start, expr.span)
            return ParsedExpression.UnaryOp(expr, op, span)
        elif self.current().variant == 'LSQUARE':
            return self.parse_array_or_dictionary_literal()
        elif self.current().variant == 'MATCH':
            return self.parse_match_expression()
        elif self.current().variant == 'LCURLY':
            return self.parse_set_literal()
        elif self.current().variant == 'AMPERSAND':
            return self.parse_ampersand()
        elif self.current().variant == 'ASTERISK':
            return self.parse_asterisk()
        elif self.current().variant == 'FUNCTION':
            return self.parse_lambda()
        else:
            span = self.current().span
            self.index_inc()
            self.error('Unsupported expression', span)
            return ParsedExpression.Invalid(span)

    def parse_set_literal(self):
        start = self.current().span
        if self.current().variant != 'LCURLY':
            self.error('Expected `{`', self.current().span)
            return ParsedExpression.Invalid(self.current().span)
        self.index_inc()

        output: List[ParsedExpression] = []
        while not self.eof():
            if self.current().variant == 'RCURLY':
                self.index_inc()
                break
            elif self.current().variant in ['COMMA', 'EOL']:
                self.index_inc()
            else:
                expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                if expr.variant == 'Invalid':
                    break
                output.append(expr)
        end = self.index - 1
        if end >= len(self.tokens) or self.tokens[end] != 'RCURLY':
            self.error('Expected `}` to close the set', self.tokens[end].span)
        return ParsedExpression.Set(output, self.merge_spans(start, self.tokens[end].span))

    def parse_match_expression(self):
        start = self.current().span
        self.index_inc()
        expr = self.parse_expression(allow_assignments=False, allow_newlines=True)
        cases = self.parse_match_cases()
        return ParsedExpression.Match(expr, cases, self.merge_spans(start, self.previous().span))

    def parse_match_cases(self):
        cases: List[ParsedMatchCase] = []

        self.skip_newlines()

        if self.current().variant != 'LCURLY':
            self.error('Expected `{`', self.current().span)
            return cases

        self.index_inc()
        self.skip_newlines()

        while not self.eof() and self.current().variant != 'RCURLY':
            pattern_start_index: int = self.index
            patterns: List[ParsedMatchPattern] = self.parse_match_patterns()

            self.skip_newlines()

            marker_span = self.current().span
            if self.current().variant == 'FAT_ARROW':
                self.index_inc()
            else:
                self.error('Expected `=>`', self.current().span)
            self.skip_newlines()

            body: ParsedMatchBody
            if self.current().variant == 'LCURLY':
                body = ParsedMatchBody.Block(self.parse_block())
            else:
                body = ParsedMatchBody.Expression(self.parse_expression(allow_assignments=False, allow_newlines=False))

            for pattern in patterns:
                cases.append(ParsedMatchCase([pattern], marker_span, body))

            if self.index == pattern_start_index:
                # parser failed to advance correctly, bail
                break
            if self.current().variant in ['COMMA', 'EOL']:
                self.index_inc()
            self.skip_newlines()
        self.skip_newlines()
        if self.current().variant == 'RCURLY':
            self.error('Expected `}`', self.current().span)
        self.index_inc()
        return cases

    def parse_match_patterns(self) -> List[ParsedMatchPattern]:
        patterns: List[ParsedMatchPattern] = []
        self.skip_newlines()
        while not self.eof():
            pattern = self.parse_match_pattern()
            patterns.append(pattern)
            self.skip_newlines()
            if self.current().variant == 'PIPE':
                self.index_inc()
                continue
            break
        return patterns

    def parse_match_pattern(self) -> ParsedMatchPattern:
        if self.current().variant in ['TRUE', 'FALSE', 'NUMBER', 'QUOTED_STRING',
                                      'SINGLE_QUOTED_STRING', 'SINGLE_QUOTED_BYTE_STRING', 'LPAREN']:
            return ParsedMatchPattern.Expression(self.parse_operand())
        elif self.current().variant == 'ELSE':
            self.index_inc()
            return ParsedMatchPattern.CatchAll()
        elif self.current().variant == 'IDENTIFIER':
            pattern_start_index = self.index
            variant_name: List[Tuple[str, TextSpan]] = []
            while not self.eof():
                if self.current().variant == 'IDENTIFIER':
                    variant_name.append((self.current().name, self.current().span))
                    self.index_inc()
                elif self.current().variant == 'COLON_COLON':
                    self.index_inc()
                else:
                    break
            variant_arguments = self.parse_variant_arguments()
            arguments_start = self.current().span
            arguments_end = self.previous().span
            arguments_span = self.merge_spans(arguments_start, arguments_end)

            return ParsedMatchPattern.EnumVariant(variant_name, variant_arguments, arguments_span)
        else:
            self.error('Expected pattern or `else`', self.current().span)
            return ParsedMatchPattern.CatchAll()

    def parse_array_or_dictionary_literal(self):
        is_dictionary: bool = False
        start = self.current().span

        if self.current().variant == 'LSQUARE':
            self.error('Expected `[`', self.current().span)
        self.index_inc()

        fill_size_expr: ParsedExpression | None = None
        output: List[ParsedExpression] = []
        dict_output: List[Tuple[ParsedExpression, ParsedExpression]] = []

        while not self.eof():
            if self.current().variant == 'RSQUARE':
                self.index_inc()
                break
            elif self.current().variant in ['COMMA', 'EOL']:
                self.index_inc()
            elif self.current().variant == 'SEMICOLON':
                if len(output) == 1:
                    self.index_inc()
                    fill_size_expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                else:
                    self.error("Can't fill and Array with more than one expression", self.current().span)
                    self.index_inc()
            elif self.current().variant == 'COLON':
                self.index_inc()
                if len(dict_output) == 0:
                    if self.current().variant == 'RSQUARE':
                        self.index_inc()
                        is_dictionary = True
                        break
                    else:
                        self.error('Expected `[`', self.current().span)
                else:
                    self.error('Missing key in dictionary literal', self.current().span)
            else:
                expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                if expr.variant == 'Invalid':
                    break
                if self.current().variant == 'COLON':
                    if len(output) > 0:
                        self.error('Mixing dictionary and array values is not allowed', self.current().span)
                    is_dictionary = True
                    self.index_inc()
                    if self.eof():
                        self.error('Key missing value in dictionary', self.current().span)
                        return ParsedExpression.Invalid(self.current().span)
                    value = self.parse_expression(allow_assignments=False, allow_newlines=False)
                    dict_output.append((expr, value))
                elif not is_dictionary:
                    output.append(expr)
        end = self.index - 1
        if end >= len(self.tokens) or self.tokens[end].variant != 'RSQUARE':
            self.error('Expected `]` to close the array', self.tokens[end].span)
        if is_dictionary:
            return ParsedExpression.Dictionary(
                    values_=dict_output,
                    span=self.merge_spans(start, self.tokens[end].span))
        else:
            return ParsedExpression.Array(values_=output, fill_size=fill_size_expr,
                                          span=self.merge_spans(start, self.tokens[end].span))

    def parse_captures(self):
        captures: List[ParsedCapture] = []
        if self.current().variant != 'LSQUARE':
            return []
        self.index_inc()
        while not self.eof():
            if self.current().variant == 'RSQUARE':
                self.index_inc()
                break
            elif self.current().variant == 'AMPERSAND':
                self.index_inc()
                if self.current().variant == 'MUT':
                    self.index_inc()
                    if self.current().variant == 'IDENTIFIER':
                        captures.append(ParsedCapture.ByMutableReference(self.current().name, self.current().span))
                        self.index_inc()
                    else:
                        self.error(f'Expected identifier, got {self.current()}', self.current().span)
                        self.index_inc()
                elif self.current().variant == 'IDENTIFIER':
                    captures.append(ParsedCapture.ByReference(self.current().name, self.current().span))
                    self.index_inc()
                else:
                    self.error(f'Expected identifier or `mut` keyword, got {self.current()}', self.current().span)
                    self.index_inc()
            elif self.current().variant == 'IDENTIFIER':
                captures.append(ParsedCapture.ByValue(self.current().name, self.current().span))
                self.index_inc()
            elif self.current().variant in ['COMMA', 'EOL']:
                self.index_inc()
            else:
                self.error(f'Unexpected token `{self.current().variant}` in captures list', self.current().span)
                self.index_inc()
        return captures

    def parse_ampersand(self):
        start = self.current().span
        self.index_inc()
        if self.current().variant == 'RAW':
            self.index_inc()
            expr = self.parse_operand()
            return ParsedExpression.UnaryOp(expr, UnaryOperator.RawAddress(), self.merge_spans(start, expr.span))
        if self.current().variant == 'MUT':
            self.index_inc()
            expr = self.parse_operand()
            return ParsedExpression.UnaryOp(expr, UnaryOperator.MutableReference(), self.merge_spans(start, expr.span))
        expr = self.parse_operand()
        return ParsedExpression.UnaryOp(expr, UnaryOperator.Reference(), self.merge_spans(start, expr.span))

    def parse_asterisk(self):
        start = self.current().span
        self.index_inc()
        expr = self.parse_operand()
        return ParsedExpression.UnaryOp(expr, UnaryOperator.Dereference(), self.merge_spans(start, self.current().span))

    def parse_lambda(self):
        start = self.current().span
        self.index_inc()
        captures = self.parse_captures()
        params = self.parse_function_parameters()
        can_throw = self.current().variant == 'THROWS'
        if can_throw:
            self.index_inc()
        return_type = self.parse_typename() if self.current().variant == 'ARROW' else ParsedType.Empty(
            self.empty_span())
        block: ParsedBlock
        if self.current().variant == 'FATARROW':
            self.index_inc()
            expr = self.parse_expression(allow_assignments=True, allow_newlines=False)
            span = expr.span
            block = ParsedBlock([ParsedStatement.Return(expr, span)])
        else:
            block = self.parse_block()
        return ParsedExpression.Function(captures, params, can_throw,
                                         return_type, block, self.merge_spans(start, self.current().span))

    def parse_operand_postfix_operator(self, start: TextSpan, expr: ParsedExpression) -> ParsedExpression:
        result = expr
        while True:
            if self.current().variant == 'DOTDOT':
                self.index_inc()
                to = self.parse_expression(allow_assignments=False, allow_newlines=False)
                result = ParsedExpression.Range(result, to, self.merge_spans(start, to.span))
            elif self.current().variant == 'EXCLAMATION_POINT':
                self.index_inc()
                result = ParsedExpression.ForcedUnwrap(result, self.merge_spans(start, self.previous().span))
            elif self.current().variant == 'PLUS_PLUS':
                self.index_inc()
                result = ParsedExpression.UnaryOp(result, UnaryOperator.PostIncrement(),
                                                  self.merge_spans(start, self.previous().span))
            elif self.current().variant == 'MINUS_MINUS':
                self.index_inc()
                result = ParsedExpression.UnaryOp(result, UnaryOperator.PostDecrement(),
                                                  self.merge_spans(start, self.previous().span))
            elif self.current().variant == 'AS':
                self.index_inc()
                cast_span = self.merge_spans(self.previous().span, self.current().span)
                cast = TypeCast.Fallible(ParsedType.Empty(self.empty_span()))
                if self.current().variant == 'EXCLAMATION_POINT':
                    self.index_inc()
                    cast = TypeCast.Infallible(self.parse_typename())
                elif self.current().variant == 'QUESTIONMARK':
                    self.index_inc()
                    cast = TypeCast.Fallible(self.parse_typename())
                else:
                    self.error('Invalid cast syntax', cast_span)
                span = self.merge_spans(start, self.merge_spans(cast_span, self.current().span))
                result = ParsedExpression.UnaryOp(result, UnaryOperator.TypeCast(cast), span)
            elif self.current().variant == 'IS':
                self.index_inc()
                parsed_type = self.parse_typename()
                span = self.merge_spans(start, self.current().span)
                bindings: List[EnumVariantPatternArgument] = []
                unary_operator_is = None
                if self.current().variant == 'LPAREN' and parsed_type.variant in ['NamespacedName', 'Name']:
                    bindings = self.parse_variant_arguments()
                    unary_operator_is = ParsedExpression.UnaryOp(result,
                                                                 UnaryOperator.IsEnumVariant(parsed_type, bindings),
                                                                 span)
                else:
                    unary_operator_is = ParsedExpression.UnaryOp(result, UnaryOperator.Is(parsed_type), span)
                result = unary_operator_is
            elif self.current().variant == 'COLON_COLON':
                result = self.parse_postfix_colon_colon(start, result)
            elif self.current().variant in ['QUESTION_MARK', 'DOT']:
                is_optional = self.current().variant == 'QUESTION_MARK'
                self.index_inc()
                if is_optional:
                    self.index_inc()
                    if self.current().variant != 'DOT':
                        self.error('Expected `.` after `?` for optional chaining access', self.current().span)
                if self.current().variant == 'NUMBER':
                    number = self.current().number
                    self.index_inc()
                    result = ParsedExpression.IndexedTuple(
                            result, number, is_optional, self.merge_spans(start, self.previous().span))
                elif self.current().variant == 'IDENTIFIER':
                    # struct field access or method call
                    name = self.current().name
                    self.index_inc()
                    if self.current().variant == 'LPAREN':
                        # Step backwards because parse_call() expects to start at the callee identifier
                        self.index -= 1
                        call = self.parse_call()
                        result = ParsedExpression.MethodCall(result, call, is_optional,
                                                             self.merge_spans(start, self.previous().span))
                    else:
                        result = ParsedExpression.IndexedStruct(result, name, is_optional,
                                                                self.merge_spans(start, self.current().span))
            elif self.current().variant == 'LSQUARE':
                # indexing operation
                self.index_inc()
                index_expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                if self.current().variant == 'RSQUARE':
                    self.index_inc()
                else:
                    self.error('Expected `]`', self.current().span)
                if index_expr.variant == 'Range':
                    from_ = index_expr.from_
                    to = index_expr.to
                    result = ParsedExpression.IndexedRangeExpression(result, from_, to, self.current().span)
                else:
                    result = ParsedExpression.IndexedExpression(result, index_expr,
                                                                self.merge_spans(start, self.current().span))
            else:
                break
        return result

    def parse_variant_arguments(self) -> List[EnumVariantPatternArgument]:
        variant_arguments: List[EnumVariantPatternArgument] = []
        has_parens: bool = False  # Note: this is weird, and maybe unnecessary
        if self.current().variant == 'LPAREN':
            has_parens = True
            self.index_inc()
            while not self.eof():
                if self.current().variant == 'IDENTIFIER':
                    arg_name = self.current().name
                    if self.peek().variant == 'COlON':
                        self.index_inc(2)
                        if self.current().variant == 'IDENTIFIER':
                            arg_binding = self.current().name
                            span = self.current().span
                            self.index_inc()
                            variant_arguments.append(EnumVariantPatternArgument(
                                    name=arg_name,
                                    binding=arg_binding,
                                    span=span))
                        else:
                            self.error('Expected binding after `:`', self.current().span)
                    else:
                        variant_arguments.append(EnumVariantPatternArgument(
                                name=None,
                                binding=arg_name,
                                span=self.current().span))
                elif self.current().variant == 'COMMA':
                    self.index_inc()
                elif self.current().variant == 'RPAREN':
                    self.index_inc()
                    break
                else:
                    self.error('Expected pattern argument name', self.current().span)
                    break
        return variant_arguments

    def parse_postfix_colon_colon(self, start: TextSpan, expr: ParsedExpression) -> ParsedExpression:
        namespace_: [str] = []
        current_name: str = ''
        self.index_inc()
        if expr.variant == 'Var':
            namespace_.append(expr.name)
        else:
            self.error('Expected namespace', expr.span)
        if self.eof():
            self.error('Incomplete static method call', self.current().span)
        while not self.eof():
            if self.current().variant != 'IDENTIFIER':
                self.error('Unsupported static method call', self.current().span)
                return expr
            current_name = self.current().name
            self.index_inc()
            if self.current().variant == 'LPAREN':
                self.index -= 1
                call = self.parse_call()
                call.namespace = namespace_
                return ParsedExpression.Call(call, self.merge_spans(expr.span, self.current().span))
            if self.current().variant == 'COLON_COLON':
                if self.previous().variant == 'IDENTIFIER':
                    namespace_.append(self.previous().name)
                else:
                    self.error('Expected namespace', expr.span)
                self.index_inc()
                continue
            if self.current().variant == 'LESS_THAN':
                self.index -= 1
                maybe_call = self.parse_call()
                if maybe_call is not None:
                    maybe_call.namespace = namespace_
                    return ParsedExpression.Call(maybe_call, self.merge_spans(expr.span, self.current().span))
                return ParsedExpression.Invalid(self.current().span)
            return ParsedExpression.NamespacedVar(name=current_name,
                                                  namespace=namespace_,
                                                  span=self.merge_spans(start, self.current().span))

    def parse_function(self, linkage: FunctionLinkage, visibility: Visibility, is_comptime: bool):
        parsed_function = ParsedFunction(
                name='',
                name_span=self.empty_span(),
                visibility=visibility,
                params=[],
                generic_parameters=[],
                block=ParsedBlock(stmts=[]),
                return_type=ParsedType.Empty(self.empty_span()),
                return_type_span=self.span(0, 0),
                can_throw=False,
                type=FunctionType.Normal(),
                linkage=linkage,
                must_instantiate=False,
                is_comptime=is_comptime
                )

        self.index_inc()

        if self.eof():
            self.error('incomplete function definition', self.current().span)
            return parsed_function

        if self.current().variant != 'IDENTIFIER':
            return parsed_function

        parsed_function.name = self.current().name
        parsed_function.name_span = self.current().span

        self.index_inc()

        parsed_function.generic_parameters = self.parse_generic_parameters()

        if self.eof():
            self.error('incomplete function', self.current().span)

        parsed_function.params = self.parse_function_parameters()

        can_throw = parsed_function.name == 'main'
        if self.current().variant == 'THROWS':
            can_throw = True
            self.index_inc()
        parsed_function.can_throw = can_throw
        if self.current().variant == 'ARROW':
            self.index_inc()
            start = self.current().span
            parsed_function.return_type = self.parse_typename()
            parsed_function.return_type_span = self.merge_spans(start, self.previous().span)

        if linkage.variant == 'External':
            return parsed_function

        if self.current().variant == 'FAT_ARROW':
            parsed_function.block = self.parse_fat_arrow()
        else:
            parsed_function.block = self.parse_block()

        return parsed_function

    def parse_fat_arrow(self):
        self.index_inc()
        start = self.current().span
        expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
        return_statement = ParsedStatement.Return(expr, self.merge_spans(start, self.current().span))
        return ParsedBlock(stmts=[return_statement])

    def parse_function_parameters(self):
        if self.current().variant == 'LPAREN':
            self.index_inc()
        else:
            self.error('Expected `(`', self.current().span)

        self.skip_newlines()

        params: List[ParsedParameter] = []
        current_param_requires_label = True
        current_param_is_mutable = False

        error = False
        parameter_complete = False

        while not self.eof():
            if self.current().variant == 'RPAREN':
                self.index_inc()
                break
            elif self.current().variant in ['COMMA', 'EOL']:
                if not parameter_complete and not error:
                    self.error('Expected parameter', self.current().span)
                    error = True
                self.index_inc()
                current_param_requires_label = True
                current_param_is_mutable = False
                parameter_complete = False
            elif self.current().variant == 'ANON':
                if parameter_complete and not error:
                    self.error('`anon` must appear at the start of parameter declaration, not the end',
                               self.current().span)
                    error = True
                if current_param_is_mutable and not error:
                    self.error('`anon` must appear before `mut`', self.current().span)
                    error = True
                if not current_param_requires_label and not error:
                    self.error('`anon` cannot appear multiple times in one parameter declaration', self.current().span)
                    error = True
                self.index_inc()
                current_param_requires_label = False
            elif self.current().variant == 'MUT':
                if parameter_complete and not error:
                    self.error('`mut` must appear at the start of parameter declaration, not the end',
                               self.current().span)
                    error = True
                if current_param_is_mutable and not error:
                    self.error('`mut` cannot appear multiple times in one parameter delcaration', self.current().span)
                    error = True
                self.index_inc()
                current_param_is_mutable = True
            elif self.current().variant == 'THIS':
                params.append(ParsedParameter(
                        requires_label=False,
                        variable=ParsedVariable(
                                name='this',
                                parsed_type=ParsedType.Empty(self.empty_span()),
                                is_mutable=current_param_is_mutable,
                                span=self.current().span
                                ),
                        default_argument=None,
                        span=self.current().span
                        ))
                self.index_inc()
                parameter_complete = True
            elif self.current().variant == 'IDENTIFIER':
                var_decl = self.parse_variable_declaration(is_mutable=current_param_is_mutable)
                default_argument = None
                if self.current().variant == 'EQUAL':
                    self.index_inc()
                    default_argument = self.parse_expression(allow_assignments=False, allow_newlines=True)
                params.append(ParsedParameter(
                        requires_label=current_param_requires_label,
                        variable=ParsedVariable(
                                name=var_decl.name,
                                parsed_type=var_decl.parsed_type,
                                is_mutable=var_decl.is_mutable,
                                span=self.previous().span
                                ),
                        default_argument=default_argument,
                        span=self.previous().span
                        ))
                parameter_complete = True
            else:
                if not error:
                    self.error('Expected parameter', self.current().span)
                    error = True
                self.index_inc()
        return params

    def parse_type_shorthand(self):
        if self.current().variant == 'LSQUARE':
            return self.parse_type_shorthand_array_or_dictionary()
        elif self.current().variant == 'LCURLY':
            return self.parse_type_shorthand_set()
        elif self.current().variant == 'LPAREN':
            return self.parse_type_shorthand_tuple()
        else:
            return ParsedType.Empty(self.empty_span())

    def parse_type_shorthand_array_or_dictionary(self):
        start = self.current().span
        self.index_inc()
        inner = self.parse_typename()
        if self.current().variant == 'RSQUARE':
            self.index_inc()
            return ParsedType.Array(inner, self.merge_spans(start, self.previous().span))
        if self.current().variant == 'COLON':
            self.index_inc()
            value = self.parse_typename()
            if self.current().variant == 'RSQUARE':
                self.index_inc()
            else:
                self.error('Expected `]`', self.current().span)
            return ParsedType.Dictionary(inner, value, self.merge_spans(start, self.current().span))
        self.error('Expected shorthand type', self.current().span)
        return ParsedType.Empty(self.empty_span())

    def parse_type_shorthand_set(self):
        start = self.current().span
        if self.current().variant == 'LCURLY':
            self.index_inc()
        inner = self.parse_typename()
        if self.current().variant == 'RCURLY':
            self.index_inc()
            return ParsedType.Set(inner, self.merge_spans(start, self.current().span))
        self.error('Expected `}`', self.current().span)
        return ParsedType.Empty(self.empty_span())

    def parse_type_shorthand_tuple(self):
        start = self.current().span
        self.index_inc()
        types: List[ParsedType] = []
        while not self.eof():
            if self.current().variant == 'RPAREN':
                self.index_inc()
                return ParsedType.Tuple(types, self.merge_spans(start, self.previous().span))
            if self.current().variant == 'COMMA':
                self.index_inc()
            index_before = self.index
            type_ = self.parse_typename()
            index_after = self.index
            if index_before == index_after:
                break
            types.append(type_)
        self.error('Expected `(`', self.current().span)
        return ParsedType.Empty(self.empty_span())

    def parse_block(self):
        start = self.current().span
        block = ParsedBlock([])

        if self.eof():
            self.error('Incomplete block', start)
            return block

        self.skip_newlines()

        if self.current().variant == 'LCURLY':
            self.index_inc()
        else:
            self.error('Expected `(`', self.current().span)

        while not self.eof():
            if self.current().variant == 'RCURLY':
                self.index_inc()
                return block
            elif self.current().variant in ['SEMICOLON', 'EOL']:
                self.index_inc()
            else:
                block.stmts.append(self.parse_statement(inside_block=True))
        self.error('Expected complete block', self.current().span)
        return block

    def parse_variable_declaration(self, is_mutable: bool):
        start = self.current().span

        if self.current().variant != 'IDENTIFIER':
            return ParsedVarDecl(
                    name='',
                    parsed_type=ParsedType.Empty(self.empty_span()),
                    is_mutable=False,
                    inlay_span=None,
                    span=start
                    )
        name = self.current().name

        self.index_inc()
        if self.current().variant == 'COLON':
            self.index_inc()
        else:
            return ParsedVarDecl(
                    name=name,
                    parsed_type=ParsedType.Empty(self.empty_span()),
                    is_mutable=is_mutable,
                    inlay_span=start,
                    span=start
                    )
        parsed_type = self.parse_typename()
        if is_mutable and parsed_type.variant in ['Reference', 'MutableReference']:
            self.error('Reference parameter can not be mutable', start)
        return ParsedVarDecl(
                name=name,
                parsed_type=parsed_type,
                is_mutable=is_mutable,
                inlay_span=None,
                span=start
                )

    def parse_statement(self, inside_block: bool):
        start = self.current().span
        if self.current().variant == 'CPP':
            self.index_inc()
            return ParsedStatement.InlineCpp(self.parse_block(), self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'DEFER':
            self.index_inc()
            statement = self.parse_statement(inside_block=False)
            return ParsedStatement.Defer(statement, self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'UNSAFE':
            self.index_inc()
            block = self.parse_block()
            return ParsedStatement.UnsafeBlock(block, self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'BREAK':
            self.index_inc()
            return ParsedStatement.Break(start)
        elif self.current().variant == 'CONTINUE':
            self.index_inc()
            return ParsedStatement.Continue(start)
        elif self.current().variant == 'LOOP':
            self.index_inc()
            block = self.parse_block()
            return ParsedStatement.Loop(block, self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'THROW':
            self.index_inc()
            expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
            return ParsedStatement.Throw(expr, self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'WHILE':
            self.index_inc()
            condition = self.parse_expression(allow_assignments=False, allow_newlines=True)
            block = self.parse_block()
            return ParsedStatement.While(condition, block, self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'YIELD':
            self.index_inc()
            expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
            if not inside_block:
                self.error('`yield` can only be used inside a block', self.merge_spans(start, expr.span))
            return ParsedStatement.Yield(expr, self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'RETURN':
            self.index_inc()
            if self.current().variant in ['EOL', 'EOF', 'RCURLY']:
                return ParsedStatement.Return(None, self.current().span)
            else:
                return ParsedStatement.Return(self.parse_expression(allow_assignments=False, allow_newlines=False),
                                              self.merge_spans(start, self.previous().span))
        elif self.current().variant in ['LET', 'MUT']:
            is_mutable = self.current().variant == 'MUT'
            self.index_inc()
            vars_: List[ParsedVarDecl] = []
            is_destructuring_assignment = False
            tuple_var_name = ''
            tuple_var_decl = ParsedVarDecl(
                    name='',
                    parsed_type=ParsedType.Empty(self.empty_span()),
                    is_mutable=is_mutable,
                    inlay_span=None,
                    span=self.current().span
                    )
            if self.current().variant == 'LPAREN':
                vars_ = self.parse_destructuring_assignment(is_mutable)
                for var in vars_:
                    tuple_var_name += var.name
                    tuple_var_name += '_'
                tuple_var_decl.name = tuple_var_name
                is_destructuring_assignment = True
            else:
                tuple_var_decl = self.parse_variable_declaration(is_mutable)
            init: ParsedExpression
            if self.current().variant == 'EQUAL':
                self.index_inc()
                init = self.parse_expression(allow_assignments=False, allow_newlines=False)
            else:
                self.error('expected initializer', self.current().span)
                init = ParsedExpression.Invalid(self.current().span)
            return_statement = ParsedStatement.VarDecl(
                    tuple_var_decl,
                    init,
                    self.merge_spans(start, self.previous().span))
            if is_destructuring_assignment:
                old_return_statement = return_statement
                return_statement = ParsedStatement.DestructuringAssignment(
                        vars_,
                        old_return_statement,
                        self.merge_spans(start, self.previous().span))
            return return_statement
        elif self.current().variant == 'IF':
            return self.parse_if_statement()
        elif self.current().variant == 'FOR':
            return self.parse_for_statement()
        elif self.current().variant == 'LCURLY':
            block = self.parse_block()
            return ParsedStatement.Block(block, self.merge_spans(start, self.previous().span))
        elif self.current().variant == 'GUARD':
            return self.parse_guard_statement()
        else:
            expr = self.parse_expression(allow_assignments=True, allow_newlines=False)
            return ParsedStatement.Expression(expr, self.merge_spans(start, self.previous().span))

    def parse_destructuring_assignment(self, is_mutable: bool) -> List[ParsedVarDecl]:
        self.index_inc()
        var_declarations: List[ParsedVarDecl] = []

        while not self.eof():
            if self.current().variant == 'IDENTIFIER':
                var_declarations.append(self.parse_variable_declaration(is_mutable))
            elif self.current().variant == 'COMMA':
                self.index_inc()
            elif self.current().variant == 'RPAREN':
                self.index_inc()
                return var_declarations
            else:
                self.error('Expected colse of destructuring assignment block', self.current().span)
                return []

    def parse_guard_statement(self):
        start = self.current().span
        self.index_inc()
        expr = self.parse_expression(allow_assignments=False, allow_newlines=True)
        if self.current().variant == 'ELSE':
            self.index_inc()
        else:
            self.error('Expected `else` keyword', self.current().span)
        else_block = self.parse_block()
        remaining_code = ParsedBlock([])
        while not self.eof():
            if self.current().variant == 'RCURLY':
                return ParsedStatement.Guard(expr, else_block, remaining_code, start)
            elif self.current().variant in ['SEMICOLON', 'EOL']:
                self.index_inc()
            else:
                remaining_code.stmts.append(self.parse_statement(inside_block=True))
        return ParsedStatement.Guard(expr, else_block, remaining_code, start)

    def parse_struct(self, definition_linkage: DefinitionLinkage):
        parsed_struct = ParsedRecord(
                name='',
                name_span=self.empty_span(),
                generic_parameters=[],
                definition_linkage=definition_linkage,
                methods=[],
                record_type=RecordType.Invalid()
                )
        if self.current().variant == 'STRUCT':
            self.index_inc()
        else:
            self.error('Expected `struct` keyword', self.current().span)
            return parsed_struct
        if self.eof():
            self.error('Incomplete struct definition, expected name', self.current().span)
            return parsed_struct
        if self.current().variant == 'IDENTIFIER':
            parsed_struct.name = self.current().name
            parsed_struct.name_span = self.current().span
            self.index_inc()
        else:
            self.error('Incomplete struct definition, expected name', self.current().span)
        if self.eof():
            self.error('Incomplete struct definition, expected generic parameters or body', self.current().span)
            return parsed_struct
        parsed_struct.generic_parameters = self.parse_generic_parameters()
        self.skip_newlines()
        if self.eof():
            self.error('Incomplete struct definition, expected body', self.current().span)
            return parsed_struct
        fields_methods = self.parse_struct_class_body(definition_linkage,
                                                      default_visibility=Visibility.Public(), is_class=False)
        parsed_struct.methods = fields_methods[1]
        parsed_struct.record_type = RecordType.Struct(fields_methods[0])
        return parsed_struct

    def parse_for_statement(self):
        start = self.current().span
        self.index_inc()

        if self.current().variant != 'IDENTIFIER':
            self.error('Expected iterator name', self.current().span)
            return ParsedStatement.Invalid(self.merge_spans(start, self.current().span))

        iterator_name = self.current().name
        name_span = self.current().span
        self.index_inc()
        if self.current().variant == 'IN':
            self.index_inc()
        else:
            self.error('Expected `in`', self.current().span)
            return ParsedStatement.Invalid(self.merge_spans(start, self.current().span))

        range_ = self.parse_expression(allow_assignments=False, allow_newlines=False)
        block = self.parse_block()

        return ParsedStatement.For(iterator_name, name_span, range_,
                                   block, self.merge_spans(start, self.previous().span))

    def parse_if_statement(self):
        if self.current().variant != 'IF':
            self.error('Expected `if` statement', self.current().span)
            return ParsedStatement.Invalid(self.current().span)

        start = self.current().span
        self.index_inc()

        condition = self.parse_expression(allow_assignments=False, allow_newlines=True)
        then_block = self.parse_block()

        else_statement: ParsedStatement | None = None
        self.skip_newlines()

        if self.current().variant == 'ELSE':
            self.index_inc()
            self.skip_newlines()
            if self.current().variant == 'IF':
                else_statement = self.parse_if_statement()
            elif self.current().variant == 'LCURLY':
                block = self.parse_block()
                if block == else_statement:
                    self.error('if and else have identical blocks', self.current().span)
                else_statement = ParsedStatement.Block(block, self.merge_spans(start, self.previous().span))
            else:
                self.error('`else` missing `if` or block', self.previous().span)
        return ParsedStatement.If(condition, then_block, else_statement, self.merge_spans(start, self.previous().span))

    def parse_typename(self):
        start = self.current().span
        is_reference = False
        is_mutable_reference = False

        if self.current().variant == 'AMPERSAND':
            is_reference = True
            self.index_inc()
            if self.current().variant == 'MUT':
                is_mutable_reference = True
                self.index_inc()
        parsed_type = self.parse_type_shorthand()
        if parsed_type.variant == 'Empty':
            parsed_type = self.parse_type_longhand()
        if self.current().variant == 'QUESTIONMARK':
            self.index_inc()
            span = self.merge_spans(start, self.current().span)
            parsed_type = ParsedType.Optional(parsed_type, span)
        if is_reference:
            span = self.merge_spans(start, self.current().span)
            if is_mutable_reference:
                parsed_type = ParsedType.MutableReference(parsed_type, span)
            else:
                parsed_type = ParsedType.Reference(parsed_type, span)
        return parsed_type

    def parse_type_longhand(self):
        parsed_type: ParsedType = ParsedType.Empty(self.empty_span())
        if self.current().variant == 'RAW':
            start = self.current().span
            self.index_inc()
            inner = self.parse_typename()
            span = self.merge_spans(start, self.current().span)
            if inner.variant == 'Optional':
                parsed_type = ParsedType.Optional(ParsedType.RawPointer(inner.inner, span), span)
            else:
                parsed_type = ParsedType.RawPointer(inner, span)
        elif self.current().variant == 'WEAK':
            start = self.current().span
            self.index_inc()
            inner = self.parse_typename()
            span = self.merge_spans(start, self.current().span)
            if inner.variant == 'Optional':
                parsed_type = ParsedType.Optional(ParsedType.WeakPointer(inner.inner, span), span)
            else:
                self.error('Missing `?` after weak ponter type name', span)
                parsed_type = ParsedType.WeakPointer(inner, span)
        elif self.current().variant == 'IDENTIFIER':
            name = self.current().name
            span = self.current().span
            self.index_inc()
            parsed_type = ParsedType.Name(name, span)
            if self.current().variant == 'LESS_THAN':
                params: List[ParsedType] = []
                if self.current().variant == 'LESS_THAN':
                    self.index_inc()
                    while self.current().variant != 'GREATER_THAN' and not self.eof():
                        params.append(self.parse_typename())
                        if self.current().variant == 'COMMA':
                            self.index_inc()
                    if self.current().variant == 'GREATER_THAN':
                        self.index_inc()
                    else:
                        self.error('Expected `>` after type parameters', self.current().span)
                parsed_type = ParsedType.GenericType(name, params, span)
            elif self.current().variant == 'COLON_COLON':
                self.index_inc()
                namespaces: List[str] = [name]
                while not self.eof():
                    if self.current().variant == 'IDENTIFIER':
                        namespace_name = self.current().name
                        if self.previous().variant == 'COLON_COLON':
                            namespaces.append(namespace_name)
                            self.index_inc()
                    elif self.current().variant == 'COLON_COLON':
                        if self.previous().variant == 'IDENTIFIER':
                            self.index_inc()
                        else:
                            self.error('Expected name after', span)
                    else:
                        break
                type_name = namespaces.pop()
                params: List[ParsedType] = []
                if self.current().variant == 'LESS_THAN':
                    self.index_inc()
                    while self.current().variant != 'GREATER_THAN' and not self.eof():
                        params.append(self.parse_typename())
                        if self.current().variant == 'COMMA':
                            self.index_inc()
                    if self.current().variant == 'GREATER_THAN':
                        self.index_inc()
                    else:
                        self.error('Expected `>` after type parameters', self.current().span)
                parsed_type = ParsedType.NamespacedName(type_name, namespaces, params, self.previous().span)
        elif self.current().variant == 'FUNCTION':
            start = self.current().span
            self.index_inc()
            params: List[ParsedParameter] = self.parse_function_parameters()
            can_throw = self.current().variant == 'THROWS'
            if can_throw:
                self.index_inc()
            return_type = ParsedType.Empty(self.empty_span())
            if self.current().variant == 'ARROW':
                self.index_inc()
                return_type = self.parse_typename()
            else:
                self.error('Expected `->` after function', self.current().span)
            parsed_type = ParsedType.Function(params, can_throw, return_type, self.merge_spans(start, return_type.span))
        else:
            self.error('Expected type name', self.current().span)
        return parsed_type

    def parse_call(self) -> ParsedCall | None:
        call = ParsedCall(
                namespace=[],
                name='',
                args=[],
                type_args=[]
                )
        if self.current().variant != 'IDENTIFIER':
            self.error('Expected Function call', self.current().span)
            return call

        call.name = self.current().name
        self.index_inc()

        index_reset = self.index

        if self.current().variant == 'LESS_THAN':
            self.index_inc()
            inner_types: List[ParsedType] = []
            while not self.eof():
                match self.current().variant:
                    case 'GREATER_THAN':
                        self.index_inc()
                        break
                    case 'COMMA':
                        self.index_inc()
                    case 'EOL':
                        self.index_inc()
                    case _:
                        index_before = self.index
                        inner_type = self.parse_typename()
                        if index_before == self.index:
                            self.index = index_reset
                            break
                        inner_types.append(inner_type)
            call.type_args = inner_types

        if self.current().variant == 'LPAREN':
            self.index_inc()
        else:
            self.index = index_reset
            self.error('Expected `(`', self.current().span)
            return None

        while not self.eof():
            match self.current().variant:
                case 'RPAREN':
                    self.index_inc()
                    break
                case 'COMMA':
                    self.index_inc()
                case 'EOL':
                    self.index_inc()
                case _:
                    label_span = self.current().span
                    label = self.parse_argument_label()

                    expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                    call.args.append((label, label_span, expr))
        return call

    def parse_argument_label(self) -> str:
        if self.peek().variant == 'COLON' and self.current().variant == 'IDENTIFIER':
            name = self.current().name
            self.index_inc(2)
            return name
        return ''

    def parse_import(self, parent: ParsedNamespace):
        if self.current().variant == 'EXTERN':
            self.index_inc()
            parent.add_extern_import(self.parse_extern_import(parent))
        else:
            parent.add_module_import(self.parse_module_import())

    def parse_extern_import(self, parent: ParsedNamespace) -> ParsedExternImport:
        parsed_import = ParsedExternImport(
                is_c=False,
                assigned_namespace=ParsedNamespace(
                        name=None,
                        name_span=None,
                        functions=[],
                        records=[],
                        namespaces=[],
                        module_imports=[],
                        extern_imports=[],
                        import_path_if_extern=None))

        if self.current().variant == 'IDENTIFIER':
            name = self.current().name
            self.index_inc()
            if name.casefold() == 'c'.casefold():
                parsed_import.is_c = True
            else:
                self.error('Expected `c` or path after `import extern`', self.current().span)
        import_path: str = ''
        if self.current().variant == 'QUOTED_STRING':
            import_path = self.current().quote
            self.index_inc()
        else:
            self.error('Expected path after `import extern`', self.current().span)

        if self.current().variant == 'AS':
            self.index_inc()
            if self.current().variant == 'IDENTIFIER':
                parsed_import.name = self.current().name
                parsed_import.name_span = self.current().span
                self.index_inc()
            else:
                self.error('Expected name after `as` keyword to name the extern import', self.current().span)

        if self.current().variant != 'LCURLY':
            self.error('Expected `{` to start namespace for the extern import', self.current().span)

        self.index_inc()

        parsed_import.assigned_namespace = self.parse_namespace()
        parsed_import.assigned_namespace.import_path_if_extern = import_path
        if self.current().variant == 'RCURLY':
            self.index_inc()
        parent.add_child_namespace(parsed_import.assigned_namespace)
        return parsed_import

    def parse_module_import(self) -> ParsedModuleImport:
        parsed_import = ParsedModuleImport(
                module_name=ImportName('', self.empty_span()),
                alias_name=None,
                import_list=[])
        if self.current().variant == 'IDENTIFIER':
            parsed_import.module_name = ImportName(self.current().name, self.current().span)
        else:
            self.error('Expected module name', self.current().span)
            return parsed_import
        self.index_inc()
        if self.eol():
            return parsed_import
        if self.current().variant == 'AS':
            self.index_inc()
            if self.current().variant == 'IDENTIFIER':
                parsed_import.alias_name = ImportName(self.current().name, self.current().span)
                self.index_inc()
            else:
                self.error('Expected name', self.current().span)
                self.index_inc()
        if self.eol():
            return parsed_import
        if self.current().variant != 'LCURLY':
            self.error('Expected `{`', self.current().span)
        self.index_inc()
        while not self.eof():
            if self.current().variant == 'IDENTIFIER':
                parsed_import.import_list.append(ImportName(self.current().name, self.current().span))
            elif self.current().variant in ['COMMA', 'EOL']:
                self.index_inc()
            elif self.current().variant == 'RCURLY':
                self.index_inc()
                break
            else:
                self.error('Expected import symbol', self.current().span)
                self.index_inc()
        return parsed_import

    def parse_enum(self, linkage: DefinitionLinkage, is_boxed: bool):
        parsed_enum = ParsedRecord(
                name='',
                name_span=self.empty_span(),
                generic_parameters=[],
                definition_linkage=linkage,
                methods=[],
                record_type=RecordType.Invalid())
        underlying_type: ParsedType | None = None
        if self.current().variant == 'ENUM':
            self.index_inc()
        else:
            self.error('Expected `enum` keyword', self.current().span)

        if self.eof():
            self.error('Incomplete enum definition, expected name', self.current().span)
            return parsed_enum

        if self.current().variant == 'IDENTIFIER':
            parsed_enum.name = self.current().name
            parsed_enum.name_span = self.current().span
            self.index_inc()
        else:
            self.error('Incomplete enum definition, expected name', self.current().span)

        if self.eof():
            self.error('Incomplete enum definition, expected generic parameters or underlying type or body',
                       self.current().span)
            return parsed_enum

        if self.current().variant == 'LESS_THAN':
            parsed_enum.generic_parameters = self.parse_generic_parameters()

        if self.eof():
            self.error("Incomplete enum definition, expected underlying type or body", self.current().span)
            return parsed_enum

        if self.current().variant == 'COLON':
            if is_boxed:
                self.error('Invalid enum definition: value enums must not have an underlying type', self.current().span)
            self.index_inc()
            underlying_type = self.parse_typename()

        self.skip_newlines()

        if self.eof():
            self.error('Incomplete enum definition, expected body', self.current().span)
            return parsed_enum

        if underlying_type:
            variants_methods = self.parse_value_enum_body(parsed_enum, linkage)
            parsed_enum.methods = variants_methods[1]
            parsed_enum.record_type = RecordType.ValueEnum(
                    underlying_type=underlying_type,
                    variants=variants_methods[0])
        else:
            variants_methods = self.parse_sum_enum_body(parsed_enum, linkage)
            parsed_enum.methods = variants_methods[1]
            parsed_enum.record_type = RecordType.SumEnum(
                    is_boxed=is_boxed,
                    variants=variants_methods[0])
        return parsed_enum

    def parse_value_enum_body(self, partial_enum: ParsedRecord, linkage: DefinitionLinkage):
        methods: List[ParsedMethod] = []
        variants: [ValueEnumVariant] = []

        if self.current().variant == 'LCURLY':
            self.index_inc()
        else:
            self.error('Expected `{` to start the enum body', self.current().span)

        self.skip_newlines()

        if self.eof():
            self.error('Incomplete enum definition, expected variant name', self.previous().span)
            return variants, methods

        last_visibility: Visibility | None = None
        last_visibility_span: TextSpan | None = None
        while not self.eof():
            if self.current().variant == 'IDENTIFIER':
                if self.peek().variant == 'EQUAL':
                    self.index_inc(2)
                    expr = self.parse_expression(allow_assignments=False, allow_newlines=False)
                    variants.append(ValueEnumVariant(self.current().name, self.current().span, expr))
                else:
                    variants.append(ValueEnumVariant(self.current().name, self.current().span, None))
                    self.index_inc()
            elif self.current().variant == 'RCURLY':
                self.index_inc()
                break
            elif self.current().variant in ['COMMA', 'EOL']:
                self.index_inc()
            elif self.current().variant == 'PRIVATE':
                if last_visibility:
                    self.error('Multiple visibility modifiers on one field or method are not allowed',
                               self.current().span)
                    last_visibility = Visibility.Private()
                    last_visibility_span = self.current().span
                    self.index_inc()
            elif self.current().variant == 'PUBLIC':
                if last_visibility:
                    self.error('Multiple visibility modifiers on one field or method are not allowed',
                               self.current().span)
                    last_visibility = Visibility.Public()
                    last_visibility_span = self.current().span
                    self.index_inc()
            elif self.current().variant in ['FUNCTION', 'COMPTIME']:
                is_comptime = self.current().variant == 'COMPTIME'
                function_linkage = FunctionLinkage.External() if linkage.variant == 'External' else FunctionLinkage.Internal()

                if function_linkage.variant == 'External' and is_comptime:
                    self.error('External functions cannot be comptime', self.current().span)

                visibility = last_visibility if last_visibility else Visibility.Public()
                last_visibility = None
                last_visibility_span = None

                parsed_method = self.parse_method(function_linkage, visibility, is_comptime)
                methods.append(parsed_method)
            else:
                self.error('Expected identifier or the end of enum block', self.current().span)
                self.index_inc()

        if self.eof():
            self.error('Invalid enum definition, expected `}`', self.current().span)
            return variants, methods

        if len(variants) == 0:
            self.error('Empty enums are not allowed', partial_enum.name_span)
        return variants, methods

    def parse_sum_enum_body(self, partial_enum: ParsedRecord, linkage: DefinitionLinkage):
        methods: List[ParsedMethod] = []
        variants: [ValueEnumVariant] = []

        if self.current().variant == 'LCURLY':
            self.index_inc()
        else:
            self.error('Expected `{` to start the enum body', self.current().span)

        self.skip_newlines()

        if self.eof():
            self.error('Incomplete enum definition, expected variant name', self.previous().span)
            return variants, methods

        last_visibility: Visibility | None = None
        last_visibility_span: TextSpan | None = None
        while not self.eof():
            if self.current().variant == 'IDENTIFIER':
                name = self.current().name
                span = self.current().span
                if self.peek().variant != 'LPAREN':
                    variants.append(SumEnumVariant(name, span, None))
                    self.index_inc()
                    continue
                self.index_inc(2)
                var_decls: List[ParsedVarDecl] = []
                while not self.eof():
                    if self.current().variant in ['IDENTIFIER', 'LSQUARE', 'LCURLY']:
                        var_decls.append(ParsedVarDecl(
                                name='',
                                parsed_type=self.parse_typename(),
                                is_mutable=False,
                                inlay_span=None,
                                span=self.current().span))
                    if self.current().variant == 'RPAREN':
                        self.index_inc()
                        break
                    elif self.current().variant in ['COMMA, EOL']:
                        self.index_inc()
                    else:
                        self.error(f'Incomplete enum variant definition, expected `,` or `)`, got {self.current()}',
                                   self.current().span)
                        break
                variants.append(SumEnumVariant(name, span, var_decls))
            elif self.current().variant == 'RCURLY':
                break
            elif self.current().variant in ['COMMA', 'EOL']:
                self.index_inc()
            elif self.current().variant == 'PRIVATE':
                if last_visibility:
                    self.error('Multiple visibility modifiers on one field or method are not allowed',
                               self.current().span)
                    last_visibility = Visibility.Private()
                    last_visibility_span = self.current().span
                    self.index_inc()
            elif self.current().variant == 'PUBLIC':
                if last_visibility:
                    self.error('Multiple visibility modifiers on one field or method are not allowed',
                               self.current().span)
                    last_visibility = Visibility.Public()
                    last_visibility_span = self.current().span
                    self.index_inc()
            elif self.current().variant in ['FUNCTION', 'COMPTIME']:
                is_comptime = self.current().variant == 'COMPTIME'
                function_linkage = FunctionLinkage.External() if linkage.variant == 'External' else FunctionLinkage.Internal()

                if function_linkage.variant == 'External' and is_comptime:
                    self.error('External functions cannot be comptime', self.current().span)

                visibility = last_visibility if last_visibility else Visibility.Public()
                last_visibility = None
                last_visibility_span = None

                parsed_method = self.parse_method(function_linkage, visibility, is_comptime)
                methods.append(parsed_method)
            else:
                self.error('Expected identifier or the end of enum block', self.current().span)
                self.index_inc()

        if self.current().variant != 'RCURLY':
            self.error('Invalid enum definition, expected `}`', self.current().span)
            return variants, methods
        self.index_inc()
        if len(variants) == 0:
            self.error('Empty enums are not allowed', partial_enum.name_span)
        return variants, methods

    def parse_record(self, definition_linkage: DefinitionLinkage) -> ParsedRecord:
        if self.current().variant == 'STRUCT':
            return self.parse_struct(definition_linkage)
        elif self.current().variant == 'CLASS':
            return self.parse_class(definition_linkage)
        elif self.current().variant == 'ENUM':
            return self.parse_enum(definition_linkage, is_boxed=False)
        elif self.current().variant == 'BOXED':
            self.index_inc()
            return self.parse_enum(definition_linkage, is_boxed=True)
        else:
            self.error('Expected `struct`, `class`, `enum`, or `boxed` keywords', self.current().span)
            return ParsedRecord(
                    name='',
                    name_span=self.empty_span(),
                    generic_parameters=[],
                    definition_linkage=definition_linkage,
                    methods=[],
                    record_type=RecordType.Invalid())

    def parse_class(self, definition_linkage: DefinitionLinkage):
        parsed_class = ParsedRecord(
                name='',
                name_span=self.empty_span(),
                generic_parameters=[],
                definition_linkage=definition_linkage,
                methods=[],
                record_type=RecordType.Invalid())
        super_class: Union[ParsedType, None] = None
        if self.current().variant == 'CLASS':
            self.index_inc()
        else:
            self.error('Expected `class` keyword', self.current().span)
            return parsed_class
        # Parse class name
        if self.eof():
            self.error('Incomplete class definition, expected name', self.current().span)
            return parsed_class
        if self.current().variant == 'IDENTIFIER':
            parsed_class.name = self.current().name
            parsed_class.name_span = self.current().span
            self.index_inc()
        else:
            self.error('Incomplete class definition, expected name', self.current().span)
        if self.eof():
            self.error('Incomplete class definition, expected generic parameters or super class or body',
                       self.current().span)
            return parsed_class
        # Parse generic parameters
        parsed_class.generic_parameters = self.parse_generic_parameters()
        if self.eof():
            self.error('Incomplete class definition, expected super class or body', self.current().span)
            return parsed_class
        # Parse super class
        if self.current().variant == 'COLON':
            self.index_inc()
            super_class = self.parse_typename()
        self.skip_newlines()
        # Parse body
        if self.eof():
            self.error('Incomplete class definition, expected body', self.current().span)
            return parsed_class
        fields_methods = self.parse_struct_class_body(
                definition_linkage, default_visibility=Visibility.Private(), is_class=True)
        parsed_class.methods = fields_methods[1]
        parsed_class.record_type = RecordType.Class(fields=fields_methods[0], super_class=super_class)
        return parsed_class

    def parse_struct_class_body(self,
                                definition_linkage: DefinitionLinkage, default_visibility: Visibility,
                                is_class: bool) -> Tuple[List[ParsedField], List[ParsedMethod]]:
        if self.current().variant == 'LCURLY':
            self.index_inc()
        else:
            self.error('Expected `{`', self.current().span)

        fields: List[ParsedField] = []
        methods: List[ParsedMethod] = []

        # gets reset after each loop. If someone doesn't consume it, we error out.
        last_visibility: Visibility | None = None
        last_visibility_span: TextSpan | None = None

        error = False

        while not self.eof():
            token_type = self.current().variant
            token_span = self.current().span
            if token_type == 'RCURLY':
                if last_visibility:
                    self.error('Expected function or parameter after visibility modifier', token_span)
                self.index_inc()
                return fields, methods
            elif token_type in ['COMMA', 'EOL']:
                self.index_inc()
            elif token_type == 'PUBLIC':
                if last_visibility:
                    self.error_with_hint('Multiple visibility modifiers on one field or method are not allowed',
                                         self.current().span, 'Previous modifier is here', last_visibility_span)
                    last_visibility = Visibility.Public()
                    last_visibility_span = self.current().span
                    self.index_inc()
            elif token_type == 'PRIVATE':
                if last_visibility:
                    self.error_with_hint('Multiple visibility modifiers on one field or method are not allowed',
                                         self.current().span, 'Previous modifier is here', last_visibility_span)
                    last_visibility = Visibility.Private()
                    last_visibility_span = self.current().span
                    self.index_inc()
            elif token_type == 'RESTRICTED':
                if last_visibility:
                    self.error_with_hint('Multiple visibility modifiers on one field or method are not allowed',
                                         self.current().span, 'Previous modifier is here', last_visibility_span)
                    last_visibility = self.parse_restricted_visibility_modifier()
                    last_visibility_span = self.current().span
                    self.index_inc()
            elif token_type == 'IDENTIFIER':
                visibility = last_visibility if last_visibility else default_visibility
                last_visibility = None
                last_visibility_span = None

                field = self.parse_field(visibility)
                fields.append(field)
            elif token_type in ['FUNCTION', 'COMPTIME']:
                is_comptime = self.current().variant == 'COMPTIME'
                function_linkage = FunctionLinkage.Internal() if definition_linkage.variant == 'Internal' else FunctionLinkage.External()
                if function_linkage.variant == 'External' and is_comptime:
                    self.error('External functions cannot be comptime', self.current().span)
                visibility = last_visibility if last_visibility else default_visibility
                last_visibility = None
                last_visibility_span = None
                parsed_method = self.parse_method(function_linkage, visibility, is_comptime=is_comptime)
                methods.append(parsed_method)
            else:
                if not error:
                    self.error(f'Invalid member, did not expect a {token_type} here', token_span)
                    error = True
                self.index_inc()
        if is_class:
            self.error('Incomplete class body, expected `}`', self.current().span)
        else:
            self.error('Incomplete struct body, expected `}`', self.current().span)
        return fields, methods

    def parse_restricted_visibility_modifier(self) -> Visibility:
        restricted_span = self.current().span

        self.index_inc()

        if self.current().variant == 'LPAREN':
            self.index_inc()
        else:
            self.error('Expected `(`', self.current().span)

        whitelist: List[ParsedType] = []
        expect_comma = False

        while not self.eof():
            if self.current().variant == 'RPAREN':
                break
            elif self.current().variant == 'COMMA':
                if expect_comma:
                    expect_comma = False
                else:
                    self.error('Unexpected comma', self.current().span)
                self.index_inc()
            else:
                if expect_comma:
                    self.error('Expected comma', self.current().span)
                self.skip_newlines()
                parsed_type = self.parse_typename()
                whitelist.append(parsed_type)
                expect_comma = True
        # Note: why not `restricted_span = self.merge_spans(restricted span, self.current().span)`?
        restricted_span.end = self.current().span.end

        if len(whitelist) == 0:
            self.error('Type list cannot be empty', restricted_span)

        if self.current().variant == 'RPAREN':
            self.index_inc()
        else:
            self.error('Expected `(`', self.current().span)

        return Visibility.Restricted(whitelist, restricted_span)

    def parse_field(self, visibility: Visibility):
        parsed_variable_declaration = self.parse_variable_declaration(is_mutable=True)

        if parsed_variable_declaration.parsed_type.variant == 'Empty':
            self.error('Field missing type', parsed_variable_declaration.span)

        return ParsedField(
                var_decl=parsed_variable_declaration,
                visibility=visibility)

    def parse_try_block(self):
        start = self.current().span
        stmt = self.parse_statement(inside_block=False)
        error_name = ''
        error_span = self.current().span

        if self.current().variant == 'CATCH':
            self.index_inc()
            if self.current().variant == 'IDENTIFIER':
                error_span = self.current().span
                error_name = self.current().name
                self.index_inc()
        else:
            self.error('Expected `catch`', self.current().span)
        catch_block = self.parse_block()
        return ParsedExpression.TryBlock(stmt, error_name, error_span,
                                         catch_block, self.merge_spans(start, self.current().span))

    def parse_method(self, linkage: FunctionLinkage, visibility: Visibility, is_comptime: bool) -> ParsedMethod:
        parsed_function = self.parse_function(linkage, visibility, is_comptime)

        if linkage.variant == 'External':
            parsed_function.must_instantiate = True

        return ParsedMethod(parsed_function, visibility)

    def parse_generic_parameters(self) -> List[ParsedGenericParameter]:
        if self.current().variant != 'LESS_THAN':
            return []
        self.index_inc()
        generic_parameters: List[ParsedGenericParameter] = []
        self.skip_newlines()
        while self.current().variant != 'GREATER_THAN' or self.current().variant != 'INVALID':
            if self.current().variant == 'IDENTIFIER':
                generic_parameters.append(ParsedGenericParameter(self.current().name, self.current().span))
                self.index_inc()
                if self.current().variant in ['COMMA', 'EOL']:
                    self.index_inc()
            else:
                self.error('Expected generic parameter name', self.current().span)
                return generic_parameters
        if self.current().variant == 'GREATER_THAN':
            self.index_inc()
        else:
            self.error('Expected `>` to end the generic parameters', self.current().span)
            return generic_parameters
        return generic_parameters

    def parse_namespace(self) -> ParsedNamespace:
        ns = ParsedNamespace(
                name=None,
                name_span=None,
                functions=[],
                records=[],
                namespaces=[],
                module_imports=[],
                extern_imports=[],
                import_path_if_extern=None
                )

        while not self.eof():
            match self.current().variant:
                case 'IMPORT':
                    self.index_inc()
                    self.parse_import(ns)
                case 'FUNCTION':
                    parsed_function = self.parse_function(FunctionLinkage.Internal(), Visibility.Public(), False)
                    ns.functions.append(parsed_function)
                case 'COMPTIME':
                    parsed_function = self.parse_function(FunctionLinkage.Internal(), Visibility.Public(), True)
                    ns.functions.append(parsed_function)
                case 'STRUCT':
                    parsed_record = self.parse_record(DefinitionLinkage.Internal())
                    ns.records.append(parsed_record)
                case 'CLASS':
                    parsed_record = self.parse_record(DefinitionLinkage.Internal())
                    ns.records.append(parsed_record)
                case 'ENUM':
                    parsed_record = self.parse_record(DefinitionLinkage.Internal())
                    ns.records.append(parsed_record)
                case 'BOXED':
                    parsed_record = self.parse_record(DefinitionLinkage.Internal())
                    ns.records.append(parsed_record)
                case 'NAMESPACE':
                    self.index_inc()
                    name: Tuple[str, TextSpan]
                    match self.current().variant:
                        case 'IDENTIFIER':
                            self.index_inc()
                            name = (self.current().name, self.current().span)
                        case _:
                            name = ('', self.empty_span())
                    if self.current().variant == 'LCURLY':
                        self.index_inc()
                    else:
                        self.error('Expected `{`', self.current().span)
                    namespace_ = self.parse_namespace()
                    if self.current().variant == 'RCURLY':
                        self.index_inc()
                    else:
                        self.error('Incomplete namespace. Are you missing a `}`?', self.previous().span)
                    if name != ():
                        namespace_.name = name[0]
                        namespace_.name_span = name[1]
                    ns.add_child_namespace(namespace_)
                case 'EXTERN':
                    self.index_inc()
                    match self.current().variant:
                        case 'FUNCTION':
                            parsed_function = self.parse_function(FunctionLinkage.External(), Visibility.Public(),
                                                                  False)
                            ns.functions.append(parsed_function)
                        case 'STRUCT':
                            parsed_struct = self.parse_struct(DefinitionLinkage.External())
                            ns.records.append(parsed_struct)
                        case 'CLASS':
                            parsed_class = self.parse_class(DefinitionLinkage.External())
                            ns.records.append(parsed_class)
                        case _:
                            self.error('Unexpected keyword', self.current().span)
                case 'EOL':
                    self.index_inc()
                case 'RCURLY':
                    break
                case _:
                    self.error('Unexpected token (expected keyword)', self.current().span)
                    break
        return ns

    def merge_spans(self, one: TextSpan, two: TextSpan):
        if two.file_id == self.compiler.current_file and two.start == 0 and two.end == 0:
            return one
        if one.file_id != two.file_id:
            self.error('Cannot merge spans between different files', one)
        return TextSpan(one.file_id, one.start, two.end)

    @classmethod
    def parse(cls, compiler: Compiler, tokens: List[Token]):
        parser = Parser(0, compiler, tokens)
        return parser.parse_namespace()
