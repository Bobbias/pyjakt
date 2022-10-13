# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

from ctypes import c_byte, c_short, c_int, c_long, c_ubyte, c_ushort, c_uint, c_ulong, c_float, c_double
from dataclasses import dataclass
from pprint import pformat
from typing import List, Dict, Any, Tuple, NoReturn, Union

from sumtype import sumtype

from compiler.compiler import Compiler
from compiler.lexing.util import panic, FileId, TextSpan
from compiler.parsedtypes import EnumVariantPatternArgument, ParsedFunction
from compiler.parsing import Visibility, DefinitionLinkage, RecordType, FunctionLinkage, FunctionType, BinaryOperator


class SafetyMode(sumtype):
    def Safe(): ...
    def Unsafe(): ...


@dataclass
class ModuleId:
    id_: int


@dataclass
class VarId:
    module: ModuleId
    id_: int


@dataclass
class FunctionId:
    module: ModuleId
    id_: int

@dataclass
class StructId:
    module: ModuleId
    id_: int


@dataclass
class EnumId:
    module: ModuleId
    id_: int


class StructOrEnumId(sumtype):
    def Struct(id_: StructId): ...
    def Enum(id_: EnumId): ...


class TypeId:
    module: ModuleId
    id_: int

    def __init__(self, module, id_):
        self.module = module
        self.id_ = id_

    @classmethod
    def none(cls):
        return None

    def to_string(self) -> str:
        return f'{self.module.id_}_{self.id_}'

    def __str__(self):
        return f'{self.module.id_}_{self.id_}'

    def __repr__(self):
        return f'TypeId(module={self.module}, id_={self.id_})'

    @classmethod
    def from_string(cls, string: str) -> TypeId | NoReturn:
        parts = string.split('_')

        if len(parts) != 2:
            panic(f'Failed to convert string `{string}` to a TypeId: Wrong number of parts.'
                  f' (Wanted 2, got {len(parts)})')

        try:
            module_id = int(parts[0])
            type_id = int(parts[1])
        except ValueError as ex:
            panic(f'Failed to convert string `{string}` to a TypeId.'
                  f' (module_id = {parts[0]}, type_id = {parts[1]}\n'
                  f'Python Exception: {ex}')

        return TypeId(ModuleId(module_id), type_id)


@dataclass
class ScopeId:
    module_id: ModuleId
    id_: int


class BuiltinType(sumtype):
    def Void(): ...
    def Bool(): ...
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
    def Usize(): ...
    def String(): ...
    def CChar(): ...
    def CInt(): ...
    def Unknown(): ...
    def Never(): ...

    def id(self):
        match self.variant:
            case 'Void': return 0
            case 'Bool': return 1
            case 'U8': return 2
            case 'U16': return 3
            case 'U32': return 4
            case 'U64': return 5
            case 'I8': return 6
            case 'I16': return 7
            case 'I32': return 8
            case 'I64': return 9
            case 'F32': return 10
            case 'F64': return 11
            case 'Usize': return 12
            case 'String': return 13
            case 'CChar': return 14
            case 'CInt': return 15
            case 'Unknown': return 16
            case 'Never': return 17


class Type(sumtype):
    def Void(): ...
    def Bool(): ...
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
    def Usize(): ...
    def String(): ...
    def CChar(): ...
    def CInt(): ...
    def Unknown(): ...
    def Never(): ...
    def TypeVariable(string: str): ...
    def GenericInstance(id_: StructId, args: List[TypeId]): ...
    def GenericEnumInstance(id_: EnumId, args: List[TypeId]): ...
    def GenericResolvedType(id_: StructId, args: List[TypeId]): ...
    def Struct(id_: StructId): ...
    def Enum(id_: EnumId): ...
    def RawPtr(id_: TypeId): ...
    def Reference(id_: TypeId): ...
    def MutableReference(id_: TypeId): ...
    def Function(params: List[TypeId], can_throw: bool, return_type_id: TypeId): ...

    def constructor_name(self):
        return self.variant

    def __eq__(self, other):
        simple_comparisons = ['Void', 'Bool',
                              'U8', 'U16', 'U32', 'U64',
                              'I8', 'I16', 'I32', 'I64',
                              'F32', 'F64',
                              'Usize', 'String', 'CChar', 'CInt']
        generics_comparisons = ['GenericInstance', 'GenericEnumInstance']
        id_comparisons = ['Struct', 'Enum', 'RawPtr', 'Reference', 'MutableReference']
        if self.variant in simple_comparisons and self.variant == other.variant:
            return True
        elif self.variant == 'TypeVariable':
            return self.variant == other.variant and self.string == other.string
        elif self.variant in generics_comparisons and self.variant == other.variant:
            same_id = self.id_ == other.id_
            same_args_len = len(self.args) == len(other.args)
            same_args_types = False
            for lhs_arg, rhs_arg in zip(self.args, other.args):
                same_args_types = same_args_types and lhs_arg == rhs_arg
            return same_id and same_args_len and same_args_types
        elif self.variant in id_comparisons and self.variant == other.variant:
            same_id = self.id_ == other.id_
            return same_id
        elif self.variant == 'Function' and self.variant == other.variant:
            same_id = self.id_ == other.id_
            same_args_len = len(self.args) == len(other.args)
            same_args_types = False
            same_return_type = self.return_type == other.return_type
            for lhs_arg, rhs_arg in zip(self.args, other.args):
                same_args_types = same_args_types and lhs_arg == rhs_arg
            return same_id and same_args_len and same_args_types and same_return_type
        else:
            return False

    def is_builtin(self) -> bool:
        return self.variant in ['Void', 'Bool',
                              'U8', 'U16', 'U32', 'U64',
                              'I8', 'I16', 'I32', 'I64',
                              'F32', 'F64',
                              'Usize', 'String', 'CChar', 'CInt']

    def get_bits(self) -> int:
        if self.variant in ['U8', 'I8', 'CChar']:
            return 8
        elif self.variant in ['U16', 'I16']:
            return 16
        elif self.variant in ['U32', 'I32', 'CInt']:
            return 32
        elif self.variant in ['U64', 'I64', 'Usize']:
            return 64
        elif self.variant == 'F32':
            return 32
        elif self.variant == 'F64':
            return 64
        else:
            return 0

    def is_signed(self) -> bool:
        if self.variant in ['I8', 'I16', 'I32', 'I64', 'CChar', 'CInt']:
            return True
        if self.variant in ['U8', 'U16', 'U32', 'U64', 'Usize']:
            return False
        if self.variant in ['F32', 'F64']:
            return True
        else:
            return False

    def min(self) -> int:
        if self.variant == 'CChar':
            return -128
        elif self.variant == 'CInt':
            return -2147483648
        elif self.variant == 'I8':
            return -128
        elif self.variant == 'I16':
            return -32768
        elif self.variant == 'I32':
            return -2147483648
        elif self.variant == 'I64':
            return -9223372036854775808
        elif self.variant.startswith('U'):
            return 0
        else:
            return 0

    def max(self) -> int:
        if self.variant == 'CChar':
            return 127
        elif self.variant == 'CInt':
            return 2147483647
        elif self.variant == 'I8':
            return 127
        elif self.variant == 'I16':
            return 32767
        elif self.variant == 'I32':
            return 2147483647
        elif self.variant == 'I64':
            return 9223372036854775807
        elif self.variant == 'U8':
            return 255
        elif self.variant == 'U16':
            return 65535
        elif self.variant == 'U32':
            return 4294967295
        elif self.variant == 'U64':
            return 18446744073709551615
        else:
            return 0

    def id(self) -> int:
        match self.variant:
            case 'Void': return 0
            case 'Bool': return 1
            case 'U8': return 2
            case 'U16': return 3
            case 'U32': return 4
            case 'U64': return 5
            case 'I8': return 6
            case 'I16': return 7
            case 'I32': return 8
            case 'I64': return 9
            case 'F32': return 10
            case 'F64': return 11
            case 'Usize': return 12
            case 'JaktString': return 13
            case 'CChar': return 14
            case 'CInt': return 15
            case 'Unknown': return 16
            case 'Never': return 17


def flip_signedness(type_: Type):
    match type_.variant:
        case 'I8': return builtin(BuiltinType.U8())
        case 'I16': return builtin(BuiltinType.U16())
        case 'I32': return builtin(BuiltinType.U32())
        case 'I64': return builtin(BuiltinType.U64)
        case 'U8': return builtin(BuiltinType.I8())
        case 'U16': return builtin(BuiltinType.I16())
        case 'U32': return builtin(BuiltinType.I32())
        case 'U64': return builtin(BuiltinType.I64())
        case _: return builtin(BuiltinType.Unknown())


def builtin(builtin_: BuiltinType) -> TypeId:
    return TypeId(ModuleId(0), builtin_.id())


@dataclass
class Scope:
    namespace_name: str | None
    vars: Dict[str, VarId]
    structs: Dict[str, StructId]
    functions: Dict[str, FunctionId]
    enums: Dict[str, EnumId]
    types: Dict[str, TypeId]
    imports: Dict[str, ModuleId]
    parent: ScopeId | None
    children: List[ScopeId]
    can_throw: bool
    import_path_if_extern: str | None
    
    debug_name: str

@dataclass
class Module:
    id_: ModuleId
    name: str
    functions: List[CheckedFunction]
    structures: List[CheckedStruct]
    enums: List[CheckedEnum]
    scopes: List[Scope]
    types: List[Type]
    variables: List[CheckedVariable]
    imports: List[ModuleId]

    is_root: bool

    def new_type_variable(self):
        new_id = len(self.types)
        self.types.append(Type.TypeVariable(f'T{new_id}'))
        return TypeId(self.id_, new_id)

    def next_function_id(self):
        return FunctionId(self.id_, len(self.functions))

    def add_function(self, checked_function: CheckedFunction):
        new_id = self.next_function_id()
        self.functions.append(checked_function)
        return new_id

    def add_variable(self, checked_variable: CheckedVariable):
        new_id = len(self.variables)
        self.variables.append(checked_variable)
        return VarId(self.id_, new_id)


@dataclass
class LoadedModule:
    module_id: ModuleId
    file_id: FileId


@dataclass
class CheckedNamespace:
    name: str
    scope: ScopeId


@dataclass
class CheckedFunction:
    name: str
    name_span: TextSpan
    visibility: Visibility
    return_type_id: TypeId
    return_type_span: TextSpan | None
    params: List[CheckedParameter]
    generic_params: List[FunctionGenericParameter]
    block: CheckedBlock
    can_throw: bool
    type: FunctionType
    linkage: FunctionLinkage
    function_scope_id: ScopeId
    is_instantiated: bool
    parsed_function: ParsedFunction | None
    is_comptime: bool

    def is_static(self):
        if len(self.params) < 1:
            return True

        return self.params[0].variable.name != 'this'

    def is_mutating(self):
        if len(self.params) < 1:
            return False

        first_param_variable = self.params[0].variable

        return first_param_variable.name == 'this' and first_param_variable.is_mutable

    def to_parsed_function(self):
        if not self.parsed_function:
            panic('to_parsed_function() called on a synthetic function')
        return self.parsed_function


@dataclass
class CheckedParameter:
    requires_label: bool
    variable: CheckedVariable
    default_value: CheckedExpression | None


class CheckedCapture(sumtype):
    def ByValue(name: str, span: TextSpan): ...
    def ByReference(name: str, span: TextSpan): ...
    def ByMutableReference(name: str, span: TextSpan): ...


class FunctionGenericParameter(sumtype):
    def InferenceGuide(id_: TypeId): ...
    def Parameter(id_: TypeId): ...


@dataclass
class CheckedVariable:
    name: str
    type_id: TypeId
    is_mutable: bool
    definition_span: TextSpan
    type_span: TextSpan | None
    visibility: Visibility

@dataclass
class CheckedVarDecl:
    name: str
    is_mutable: bool
    span: TextSpan
    type_id: TypeId


class BlockControlFlow(sumtype):
    def AlwaysReturns(): ...
    def AlwaysTransfersControl(might_break: bool): ...
    def NeverReturns(): ...
    def MayReturn(): ...
    def PartialAlwaysReturns(might_break: bool): ...
    def PartialAlwaysTransfersControl(might_break: bool): ...
    def PartialNeverReturns(might_break: bool): ...

    def unify_with(self, second):
        if self.variant == 'NeverReturns':
            return second
        elif self.variant == 'AlwaysReturns':
            if second.variant in ['NeverReturns', 'AlwaysReturns']:
                return BlockControlFlow.AlwaysReturns()
            elif second.variant == 'AlwaysTransfersControl':
                return BlockControlFlow.AlwaysTransfersControl(second.might_break)
            elif second.variant == 'MayReturn':
                return BlockControlFlow.MayReturn()
            elif second.variant == 'PartialAlwaysReturns':
                return BlockControlFlow.AlwaysReturns()
            elif second.variant == 'PartialAlwaysTransfersControl':
                return BlockControlFlow.AlwaysTransfersControl(second.might_break)
            elif second.variant == 'PartialNeverReturns':
                return BlockControlFlow.AlwaysTransfersControl(second.might_break)
        elif self.variant == 'AlwaysTransfersControl':
            if second.variant in ['NeverReturns', 'AlwaysReturns']:
                return BlockControlFlow.AlwaysTransfersControl(self.might_break)
            elif second.variant == 'AlwaysTransfersControl':
                return BlockControlFlow.AlwaysTransfersControl(self.might_break or second.might_break)
            elif second.variant == 'MayReturn':
                return BlockControlFlow.MayReturn()
            else:
                return BlockControlFlow.AlwaysTransfersControl(self.might_break)
        elif self.variant == 'MayReturn':
            if second.variant == 'PartialAlwaysReturns':
                return BlockControlFlow.PartialAlwaysReturns(second.might_break)
            elif second.variant == 'PartialAlwaysTransfersControl':
                return BlockControlFlow.PartialAlwaysTransfersControl(second.might_break)
            elif second.variant == 'PartialNeverReturns':
                return BlockControlFlow.PartialNeverReturns(second.might_break)
            else:
                return BlockControlFlow.MayReturn()
        elif self.variant == 'PartialAlwaysReturns':
            if second.variant == 'PartialAlwaysReturns':
                return BlockControlFlow.PartialAlwaysReturns(self.might_break or second.might_break)
            elif second.variant in ['PartialAlwaysTransfersControl', 'PartialNeverReturns']:
                return BlockControlFlow.PartialAlwaysReturns(self.might_break or second.might_break)
            elif second.variant == 'AlwaysReturns':
                return BlockControlFlow.AlwaysReturns()
            elif second.variant == 'NeverReturns':
                return BlockControlFlow.AlwaysTransfersControl(self.might_break)
            elif second.variant == 'AlwaysTransfersControl':
                return BlockControlFlow.AlwaysTransfersControl(self.might_break or second.might_break)
            elif second.variant == 'MayReturn':
                return BlockControlFlow.PartialAlwaysReturns(self.might_break)
        elif self.variant == 'PartialAlwaysTransfersControl':
            if second.variant in ['PartialAlwaysTransfersControl', 'PartialAlwaysReturns', 'PartialNeverReturns']:
                return BlockControlFlow.PartialAlwaysTransfersControl(self.might_break or second.might_break)
            elif second.variant in ['AlwaysReturns', 'NeverReturns']:
                return BlockControlFlow.AlwaysTransfersControl(self.might_break)
            elif second.variant == 'AlwaysTransfersControl':
                return BlockControlFlow.AlwaysTransfersControl(self.might_break or second.might_break)
            elif second.variant == 'MayReturn':
                return BlockControlFlow.PartialAlwaysTransfersControl(self.might_break)
        elif self.variant == 'PartialNeverReturns':
            if second.variant == 'PartialNeverReturns':
                return BlockControlFlow.PartialNeverReturns(self.might_break or second.might_break)
            elif second.variant in ['PartialAlwaysTransfersControl', 'PartialAlwaysReturns']:
                return BlockControlFlow.PartialAlwaysTransfersControl(self.might_break or second.might_break)
            elif second.variant == 'AlwaysReturns':
                return BlockControlFlow.AlwaysTransfersControl(self.might_break)
            elif second.variant == 'AlwaysTransfersControl':
                return BlockControlFlow.AlwaysTransfersControl(self.might_break or second.might_break)
            elif second.variant == 'MayReturn':
                return BlockControlFlow.PartialNeverReturns(self.might_break)
            elif second.variant == 'NeverReturns':
                return BlockControlFlow.NeverReturns()

    def updated(self, second):
        if self.variant == 'NeverReturns':
            return BlockControlFlow.NeverReturns()
        elif self.variant == 'AlwaysReturns':
            return BlockControlFlow.AlwaysReturns()
        elif self.variant == 'AlwaysTransfersControl':
            return BlockControlFlow.AlwaysTransfersControl(self.might_break)
        elif self.variant == 'MayReturn':
            return second
        elif self.variant in ['PartialAlwaysTaransfersControl', 'PartialAlwaysReturns', 'PartialNeverReturns']:
            return self.unify_with(second)

    def partial(self):
        if self.variant == 'NeverReturns':
            return BlockControlFlow.PartialNeverReturns(might_break=False)
        elif self.varaitn == 'AlwaysReturns':
            return BlockControlFlow.PartialAlwaysReturns(might_break=False)
        elif self.variant == 'MayReturn':
            return BlockControlFlow.MayReturn()
        elif self.variant == 'AlwaysTransfersControl':
            return BlockControlFlow.PartialAlwaysTransfersControl(self.might_break)
        elif self.variant == 'PartialAlwaysTransfersControl':
            return BlockControlFlow.PartialAlwaysTransfersControl(self.might_break)
        elif self.variant == 'PartialAlwaysReturns':
            return BlockControlFlow.PartialAlwaysReturns(self.might_break)
        elif self.variant == 'PartialNeverReturns':
            return BlockControlFlow.PartialNeverReturns(self.might_break)

    def always_transfers_control(self):
        if self.variant in ['AlwaysReturns', 'AlwaysTransfersControl']:
            return True
        else:
            return False
        
    def never_returns(self):
        if self.variant == 'NeverReturns':
            return True
        else:
            return False

    def always_returns(self):
        if self.variant == 'AlwaysReturns':
            return True
        else:
            return False

    def may_return(self):
        if self.variant in ['MayReturn', 'PartialAlwaysReturns', 'PartialAlwaysTransfersControl', 'PartialNeverReturns']:
            return True
        else:
            return False

    def may_break(self):
        if self.variant in ['PartialAlwaysReturns', 'PartialAlwaysTransfersControl', 'PartialNeverReturns', 'AlwaysTransfersControl']:
            return self.might_break
        else:
            return False

    def is_reachable(self):
        if self.variant in ['NeverReturns', 'AlwaysReturns', 'AlwaysTransfersControl']:
            return False
        else:
            return True


@dataclass
class CheckedBlock:
    statements: List[CheckedStatement]
    scope_id: ScopeId
    control_flow: BlockControlFlow
    yielded_type: TypeId | None


@dataclass
class CheckedStruct:
    name: str
    name_span: TextSpan
    generic_parameters: List[TypeId]
    fields: List[VarId]
    scope_id: ScopeId
    definition_linkage: DefinitionLinkage
    record_type: RecordType
    type_id: TypeId


@dataclass
class CheckedEnum:
    name: str
    name_span: TextSpan
    generic_parameters: List[TypeId]
    variants: List[CheckedEnumVariant]
    scope_id: ScopeId
    definition_linkage: DefinitionLinkage
    record_type: RecordType
    underlying_type_id: TypeId
    type_id: TypeId
    is_boxed: bool


class CheckedEnumVariant(sumtype):
    def Untyped(enum_id: EnumId, name: str, span: TextSpan): ...
    def Typed(enum_id: EnumId, name: str, type_id: TypeId, span: TextSpan): ...
    def WithValue(enum_id: EnumId, name: str, expr: Any, span: TextSpan): ...
    def StructLike(enum_id: EnumId, name: str, fields: List[VarId], span: TextSpan): ...

    def __eq__(self, other):
        if self.variant == 'Untyped' and self.variant == other.variant:
            return self.name == other.name
        else:
            return False


@dataclass
class CheckedEnumVariantBinding:
    name: str | None
    binding: str
    type_id: TypeId
    span: TextSpan


class CheckedStatement(sumtype):
    def Expression(expr: Any, span: TextSpan): ...
    def Defer(statement: Any, span: TextSpan): ...
    def DestructuringAssignment(vars_: List[Any], var_decl: Any, span: TextSpan): ...
    def VarDecl(var_id: VarId, init: Any, span: TextSpan): ...
    def If(condition: Any, then_block: CheckedBlock, else_statement: Any, span: TextSpan): ... # Condition: CheckedExpression, else_statement: CheckedExpression | None
    def Block(block: CheckedBlock, span: TextSpan): ...
    def Loop(block: CheckedBlock, span: TextSpan): ...
    def While(condition: Any, block: CheckedBlock, span: Union[TextSpan, None]): ...  # condition: CheckedExpression | None
    def Return(val: Any, span: TextSpan): ...  # val: CheckedExpression | None
    def Break(span: TextSpan): ...
    def Continue(span: TextSpan): ...
    def Throw(expr: Any, span: TextSpan): ...
    def Yield(expr: Any, span: TextSpan): ...
    def InlineCpp(lines: List[str], span: TextSpan): ...
    def Invalid(span: TextSpan): ...


class NumberConstant(sumtype):
    def Signed(val: Any): ...  # val: c_long
    def Unsigned(val: Any): ...  # val: c_ulong
    def Floating(val: Any): ...  # val: c_double

    def can_fit_number(self, type_id: TypeId, program: CheckedProgram) -> bool:
        type_ = program.get_type(type_id)

        if self.variant == 'Signed':
            if type_.variant == 'I64':
                return True
            elif type_.variant in ['U64', 'Usize']:
                return self.val >= 0
            else:
                return program.is_integer(type_id) and type_.min() <= self.val <= type_.max()
        elif self.variant == 'Unsigned':
            if type_.variant in ['U64', 'Usize']:
                return True
            else:
                return program.is_integer(type_id) and self.val <= type_.max()
        elif self.variant == 'Floating':
            if type_.variant == 'F32':
                # Jakt todo: Implement casting F32 to F64
                return False
            elif type_.variant == 'F64':
                return True
            else:
                return False

    def to_usize(self) -> c_ulong:
        if self.variant in ['F32', 'F64']:
            panic('to_usize on a floating point constant')
        else:
            return c_ulong(self.val)

    def promote(self, type_id: TypeId, program: CheckedProgram) -> CheckedNumericConstant | None:
        if not self.can_fit_number(type_id, program):
            return None

        bits = program.get_bits(type_id)
        is_signed = program.is_signed(type_id)
        # note: In Jakt, these are separate, likely because Signed and Unsigned have different underlying types
        # in the original source, whereas both simply store python integers currently
        result = None
        if self.variant == ['Signed', 'Unsigned']:
            if is_signed:
                match bits:
                    case 8:
                        result = CheckedNumericConstant.U8(c_ubyte(self.val.value))
                    case 16:
                        result = CheckedNumericConstant.U16(c_ushort(self.val.value))
                    case 32:
                        result = CheckedNumericConstant.U32(c_uint(self.val.value))
                    case 64:
                        result = CheckedNumericConstant.U64(c_ulong(self.val.value))
                    case _:
                        panic('Numeric constants can only be 8, 16, 32, or 64 bits long')
            else:
                match bits:
                    case 8:
                        result = CheckedNumericConstant.I8(c_byte(self.val.value))
                    case 16:
                        result = CheckedNumericConstant.I16(c_short(self.val.value))
                    case 32:
                        result = CheckedNumericConstant.I32(c_int(self.val.value))
                    case 64:
                        result = CheckedNumericConstant.I64(c_long(self.val.value))
                    case _:
                        panic('Numeric constants can only be 8, 16, 32, or 64 bits long')
        elif self.variant == 'Floating':
            if is_signed:
                match bits:
                    case 32:
                        # todo: add f64 to f32 conversion
                        result = CheckedNumericConstant.F32(c_float(0))
                    case 64:
                        result = CheckedNumericConstant.F64(c_double(self.val.value))
            else:
                panic('Floating numeric constant cannot be unsigned')

        return result


class CheckedNumericConstant(sumtype):
    def I8(val: Any): ...  # val: c_byte
    def I16(val: Any): ...  # val: c_short
    def I32(val: Any): ...  # val: c_int
    def I64(val: Any): ...  # val: c_long
    def U8(val: Any): ...  # val: c_ubyte
    def U16(val: Any): ...  # val: c_ushort
    def U32(val: Any): ...  # val: c_uint
    def U64(val: Any): ...  # val: c_ulong
    def Usize(val: Any): ...  # val: c_ulong
    def F32(val: Any): ...  # val: c_float
    def F64(val: Any): ...  # val: c_double
    
    def number_constant(self):
        if self.variant.startswith('I'):
            return NumberConstant.Signed(self.val.value)
        elif self.variant.startswith('U'):
            return NumberConstant.Unsigned(self.val.value)
        elif self.variant.startswith('F'):
            return NumberConstant.Floating(self.val.value)
        else:
            return None


class CheckedTypeCast(sumtype):
    def Fallible(id_: TypeId): ...
    def Infallible(id_: TypeId): ...


class CheckedUnaryOperator(sumtype):
    def PreIncrement(): ...
    def PostIncrement(): ...
    def PreDecrement(): ...
    def PostDecrement(): ...
    def Negate(): ...
    def Dereference(): ...
    def RawAddress(): ...
    def Reference(): ...
    def MutableReference(): ...
    def LogicalNot(): ...
    def BitwiseNot(): ...
    def TypeCast(checked_type_cast: CheckedTypeCast): ...
    def Is(type_id: TypeId): ...
    def IsEnumVariant(enum_variant: CheckedEnumVariant, bindings: List[CheckedEnumVariantBinding], type_id: TypeId): ...


class CheckedMatchBody(sumtype):
    def Expression(expr: Any): ...  # expr: CheckedExpression
    def Block(block: CheckedBlock): ...


class CheckedMatchCase(sumtype):
    def EnumVariant(name: str, args: List[EnumVariantPatternArgument], subject_type_id: TypeId, index: int, scope_id: ScopeId, body: CheckedMatchBody, marker_span: TextSpan): ...
    def Expression(expression: Any, body: CheckedMatchBody, marker_span: TextSpan): ...  # expression: CheckedExpression
    def CatchAll(body: CheckedMatchBody, marker_span: TextSpan): ...


class CheckedExpression(sumtype):
    def Boolean(val: bool, span: TextSpan): ...
    def NumericConstant(val: CheckedNumericConstant, span: TextSpan, type_id: TypeId): ...
    def QuotedString(val: str, span: TextSpan): ...
    def ByteConstant(val: str, span: TextSpan): ...
    def CharacterConstant(val: str, span: TextSpan): ...
    def UnaryOp(expr: Any, op: CheckedUnaryOperator, span: TextSpan, type_id: TypeId): ...
    def BinaryOp(lhs: Any, op: BinaryOperator, rhs: Any, span: TextSpan, type_id: TypeId): ...
    def Tuple(vals: List[Any], span: TextSpan, type_id: TypeId): ...
    def Range(from_: Any, to: Any, span: TextSpan, type_id: TypeId): ...
    def Array(vals: List[Any], repeat: Any, span: TextSpan, type_id: TypeId, inner_type_id: TypeId): ...  # repeat: CheckedExpression | None
    def Set(vals: List[Any], span: TextSpan, type_id: TypeId, inner_type_id: TypeId): ...
    def Dictionary(vals: List[Tuple[Any, Any]], span: TextSpan, type_id: TypeId, key_type_id: TypeId, value_type_id: TypeId): ...
    def IndexedExpression(expr: Any, index: Any, span: TextSpan, type_id: TypeId): ...
    def IndexedRangeExpression(expr: Any, from_: Any, to: Any, span: TextSpan, type_id: TypeId): ...
    def IndexedDictionary(expr: Any, index: Any, span: TextSpan, type_id: TypeId): ...
    def IndexedTuple(expr: Any, index: int, span: TextSpan, is_optional: bool, type_id: TypeId): ...
    def IndexedStruct(expr: Any, index: str, span: TextSpan, is_optional: bool, type_id: TypeId): ...
    def Match(expr: Any, match_cases: List[CheckedMatchCase], span: TextSpan, type_id: TypeId, all_variants_constant: bool): ...
    def EnumVariantArg(expr: Any, arg: CheckedEnumVariantBinding, enum_variant: CheckedEnumVariant, span: TextSpan): ...
    def Call(call: Any, span: TextSpan, type_id: TypeId): ...  # call: CheckedCall
    def MethodCall(expr: Any, call: Any, span: TextSpan, is_optional: bool, type_id: TypeId): ...  # call: CheckedCall
    def NamespacedVar(namespaces: List[CheckedNamespace], var: CheckedVariable, span: TextSpan): ...
    def Var(var: CheckedVariable, span: TextSpan): ...
    def OptionalNone(span: TextSpan, type_id: TypeId): ...
    def OptionalSome(expr: Any, span: TextSpan, type_id: TypeId): ...
    def ForcedUnwrap(expr: Any, span: TextSpan, type_id: TypeId): ...
    def Block(block: CheckedBlock, span: TextSpan, type_id: TypeId): ...
    def Function(captures: List[CheckedCapture], params: List[CheckedParameter], can_throw: bool, return_type_id: TypeId, block: CheckedBlock, span: TextSpan, type_id: TypeId): ...
    def Try(expr: Any, catch_block: Union[CheckedBlock, None], catch_name: Union[str, None], span: TextSpan, type_id: TypeId, inner_type_id: TypeId): ...
    def TryBlock(stmt: CheckedStatement, catch_block: CheckedBlock, error_name: str, error_span: TextSpan, span: TextSpan, type_id: TypeId): ...
    def Invalid(span: TextSpan): ...

    def to_number_constant(self, program: CheckedProgram):
        if self.variant == 'NumericConstant':
            self.val.number_constant()
        elif self.variant == 'UnaryOp':
            if self.op.variant == 'TypeCast':
                if self.op.cast.variant != 'Infallible':
                    return None
                if not program.is_integer(self.op.type_id) and not program.is_floating(self.op.type_id):
                    return None
                if self.op.expr.variant == 'NumericConstant':
                    return self.op.expr.val.number_constant()
                else:
                    return None
        return None

    def is_mutable(self, program: CheckedProgram):
        if self.variant == 'Var':
            return self.is_mutable
        elif self.variant in ['IndexedStruct', 'IndexedExpression', 'IndexedTuple', 'IndexedDictionary', 'ForcedUnwrap']:
            return self.expr.is_mutable(program)
        elif self.variant == 'UnaryOp':
            if self.op.variant == 'Dereference':
                variant = program.get_type(self.expr.type()).variant
                if variant == 'MutableReference':
                    return True
                elif variant == 'RawPtr':
                    return self.expr.is_mutable(program)
                else:
                    return False
            else:
                return False
        elif self.variant == 'MethodCall':
            return self.expr.is_mutable(program)
        else:
            return False

    def can_throw(self):
        if self.variant in ['Call', 'MethodCall']:
            return self.call.callee_throws
        else:
            return False

    def type(self):
        if self.variant == 'Boolean':
            return builtin(BuiltinType.Bool())
        if self.variant == 'NumericConstant':
            return self.type_id
        if self.variant == 'QuotedString':
            return builtin(BuiltinType.String())
        if self.variant == 'ByteConstant':
            return builtin(BuiltinType.U8())
        if self.variant == 'CharacterConstant':
            return builtin(BuiltinType.CChar())
        if self.variant == 'UnaryOp':
            return self.type_id
        if self.variant == 'BinaryOp':
            return self.type_id
        if self.variant == 'Tuple':
            return self.type_id
        if self.variant == 'Range':
            return self.type_id
        if self.variant == 'Array':
            return self.type_id
        if self.variant == 'Dictionary':
            return self.type_id
        if self.variant == 'Set':
            return self.type_id
        if self.variant == 'IndexedExpression':
            return self.type_id
        if self.variant == 'IndexedRangeExpression':
            return self.type_id
        if self.variant == 'IndexedDictionary':
            return self.type_id
        if self.variant == 'IndexedTuple':
            return self.type_id
        if self.variant == 'IndexedStruct':
            return self.type_id
        if self.variant == 'Call':
            return self.type_id
        if self.variant == 'MethodCall':
            return self.type_id
        if self.variant == 'NamespacedVar':
            return self.var.type_id
        if self.variant == 'Var':
            return self.var.type_id
        if self.variant == 'OptionalNone':
            return self.type_id
        if self.variant == 'OptionalSome':
            return self.type_id
        if self.variant == 'ForcedUnwrap':
            return self.type_id
        if self.variant == 'Match':
            return self.type_id
        if self.variant == 'EnumVariantArg':
            return self.arg.type_id
        if self.variant == 'Block':
            return self.type_id
        if self.variant == 'Function':
            return self.type_id
        if self.variant == 'Try':
            return self.type_id
        if self.variant == 'TryBlock':
            return self.type_id
        if self.variant == 'Garbage':
            return builtin(BuiltinType.Void())

    def control_flow(self):
        if self.variant == 'Match':
            control_flow: BlockControlFlow | None = None
            for case in self.match_cases:
                case_control_flow: BlockControlFlow | None = None
                if case.variant in ['EnumVariant', 'Expression', 'CatchAll']:
                    if case.body.variant == 'Block':
                        case_control_flow = case.body.block.control_flow
                    elif case.body.variant == 'Expression':
                        case_control_flow = case.body.expr.control_flow()
                if case_control_flow:
                    control_flow = control_flow.unify_with(case_control_flow)
                else:
                    control_flow = case_control_flow
            if control_flow:
                return control_flow
            else:
                return BlockControlFlow.MayReturn()
        elif self.variant in ['MethodCall', 'Call']:
            if self.type_id == never_type_id():
                return BlockControlFlow.NeverReturns()
            else:
                return BlockControlFlow.MayReturn()
        else:
            return BlockControlFlow.MayReturn()


@dataclass
class ResolvedNamespace:
    name: str
    generic_parameters: List[TypeId] | None


@dataclass
class CheckedCall:
    namespace: List[ResolvedNamespace]
    name: str
    args: List[Tuple[str, CheckedExpression]]
    type_args: List[TypeId]
    function_id: FunctionId | None
    return_type: TypeId
    callee_throws: bool


def unknown_type_id() -> TypeId: builtin(BuiltinType.Unknown())
def void_type_id() -> TypeId: builtin(BuiltinType.Void())
def never_type_id() -> TypeId: builtin(BuiltinType.Never())


@dataclass
class CheckedProgram:
    compiler: Compiler
    modules: List[Module]
    loaded_modules: Dict[str: LoadedModule]

    def __init__(self, compiler: Compiler, modules: List[Module], loaded_modules: Dict[str, LoadedModule]):
        self.compiler = compiler
        self.modules = modules
        self.loaded_modules = loaded_modules

    def get_module(self, module_id: ModuleId) -> Module:
        return self.modules[module_id.id_]

    def get_function(self, function_id: FunctionId) -> CheckedFunction:
        return self.modules[function_id.module.id_].functions[function_id.id_]

    def get_variable(self, variable_id: VarId) -> CheckedVariable:
        return self.modules[variable_id.module.id_].variables[variable_id.id_]

    def get_type(self, type_id: TypeId) -> Type:
        print(type_id)
        print(f'get_type: typeid: {type_id.id_}, module: {self.modules[type_id.module.id_].name}, types: {",".join([f"{type_.variant}({id_})" for id_, type_ in enumerate(self.modules[type_id.module.id_].types)])}, type: {self.modules[type_id.module.id_].types[type_id.id_]}')
        return self.modules[type_id.module.id_].types[type_id.id_]

    def get_enum(self, enum_id: EnumId) -> CheckedEnum:
        return self.modules[enum_id.module.id_].enums[enum_id.id_]

    def get_struct(self, struct_id: StructId) -> CheckedStruct:
        print(f'get_struct: struct_id: {struct_id.id_}, module: {self.modules[struct_id.module.id_]}, structs: {self.modules[struct_id.module.id_].structures}, struct: {self.modules[struct_id.module.id_].structures[struct_id.id_]}')
        return self.modules[struct_id.module.id_].structures[struct_id.id_]

    def get_scope(self, scope_id: ScopeId) -> Scope:
        module_id = scope_id.module_id.id_
        module = self.modules[module_id]
        print(f'Module {module_id}: {module.name}')
        max_scope = len(module.scopes) - 1
        if scope_id.id_ > max_scope:
            self.compiler.panic(f'scope_id {scope_id} does not exist in module')
        return self.modules[scope_id.module_id.id_].scopes[scope_id.id_]

    def prelude_scope_id(self) -> ScopeId:
        return ScopeId(ModuleId(0), 0)

    def set_loaded_module(self, module_name: str, loaded_module: LoadedModule):
        print(f'setting loaded module: {module_name}')
        self.loaded_modules[module_name] = loaded_module

    def get_loaded_module(self, module_name: str) -> LoadedModule:
        return self.loaded_modules.get(module_name, None)

    def find_var_in_scope(self, scope_id: ScopeId, var: str):
        current_scope_id = scope_id
        while True:
            scope = self.get_scope(current_scope_id)
            maybe_var = scope.vars.get(var, None)
            if maybe_var:
                return self.get_variable(maybe_var)
            if not scope.parent:
                break
            current_scope_id = scope.parent
        return None

    def find_enum_in_scope(self, scope_id: ScopeId, name: str):
        current_scope_id = scope_id
        while True:
            scope = self.get_scope(current_scope_id)
            maybe_enum = scope.enums.get(name, None)
            if maybe_enum:
                return maybe_enum
            for child_id in scope.children:
                child_scope = self.get_scope(child_id)
                if not child_scope.namespace_name:
                    maybe_enum = child_scope.enums.get(name, None)
                    if maybe_enum:
                        return maybe_enum
            if scope.parent:
                current_scope_id = scope.parent
            else:
                break

        return None

    def is_integer(self, type_id: TypeId):
        type = self.get_type(type_id)

        if type.variant in ['I8', 'I16', 'I32', 'I64', 'U8', 'U16', 'U32', 'U64', 'Usize', 'CInt', 'CChar']:
            return True
        else:
            return False

    def is_floating(self, type_id: TypeId):
        type = self.get_type(type_id)

        if type.variant in ['F32', 'F64']:
            return True
        else:
            return False

    def is_numeric(self, type_id: TypeId) -> bool:
        return self.is_floating(type_id) or self.is_integer(type_id)

    def is_string(self, type_id: TypeId) -> bool:
        return self.get_type(type_id).variant == 'String'

    def get_bits(self, type_id: TypeId) -> int:
        return self.get_type(type_id).get_bits()

    def is_signed(self, type_id: TypeId) -> bool:
        return self.get_type(type_id).is_signed()

    def find_struct_in_scope(self, scope_id: ScopeId, name: str):
        print(f'find_struct_in_scope: find struct {name} in {pformat(scope_id, depth=1)}:')
        current_scope_id = scope_id
        while True:
            scope = self.get_scope(current_scope_id)
            print(f'find_struct_in_scope: scope = {pformat(scope, depth=1)}')
            maybe_struct = scope.structs.get(name, None)
            if maybe_struct:
                print(f'find_struct_in_scope: found struct {name} as {maybe_struct} in {scope.debug_name}')
                return maybe_struct
            for child_id in scope.children:
                child_scope = self.get_scope(child_id)
                print(f'find_struct_in_scope: searching child scope: {pformat(child_scope, depth=1)}')
                if not child_scope.namespace_name:
                    maybe_struct = child_scope.structs.get(name, None)
                    if maybe_struct:
                        return maybe_struct
            if scope.parent:
                print(f'find_struct_in_scope: checking parent scope: {pformat(scope.parent, depth=1)}')
                current_scope_id = scope.parent
            else:
                print('find_struct_in_scope: No parent scope to check')
                break
        print(f'find_struct_in_scope: Struct `{name}` not found in scope `{pformat(scope_id, depth=1)}`')
        return None

    def find_struct_in_prelude(self, name: str):
        scope_id = self.prelude_scope_id()
        print(f'find_struct_in_prelude: name: {name} scope_id: {scope_id}')
        struct_id = self.find_struct_in_scope(scope_id, name)
        print(f'find_struct_in_prelude: struct_id: {struct_id}')
        if struct_id:
            return struct_id

        self.compiler.panic(f'Internal error: {name} builtin definition not found')

    def find_namespace_in_scope(self, scope_id: ScopeId, name: str) -> Tuple[ScopeId, bool]:
        """
        Returns a tuple containing the `ScopeId` which contains the given namespace and a boolean indicating
        whether the namespace is imported or not.

        :param scope_id:
        :type scope_id:
        :param name:
        :type name:
        :return:
        :rtype:
        """
        current_scope_id = scope_id
        while True:
            scope = self.get_scope(current_scope_id)

            for child_id in scope.children:
                child = self.get_scope(child_id)
                if child.namespace_name:
                    if name == child.namespace_name:
                        return child_id, False

            for child_id in scope.children:
                child = self.get_scope(child_id)
                if not child.namespace_name:
                    for descendant_scope_id in self.get_scope(child_id).children:
                        descendant_scope = self.get_scope(descendant_scope_id)
                        if descendant_scope.namespace_name:
                            if name == descendant_scope.namespace_name:
                                return descendant_scope_id, False

            if scope.parent:
                current_scope_id = scope.parent
            else:
                break

        module_id = scope_id.module_id

        search_scope_id = ScopeId(module_id, 0)
        search_scope = self.get_scope(search_scope_id)
        search_imports = search_scope.imports
        maybe_import = search_imports[name]
        if maybe_import:
            import_module_id = maybe_import
            import_scope_id = ScopeId(import_module_id, 0)
            return import_scope_id, True

    def find_function_in_scope(self, parent_scope_id: ScopeId, function_name: str):
        visited: List[ScopeId] = []
        queue: List[ScopeId] = [parent_scope_id]
        scope_id = parent_scope_id

        while len(queue):
            scope_id = queue.pop()
            was_visited = False
            for visited_id in visited:
                if visited_id == scope_id:
                    was_visited = True
                    break
            if was_visited:
                continue
            visited.append(scope_id)
            scope = self.get_scope(scope_id)
            maybe_function = scope.functions.get(function_name, None)
            if maybe_function:
                return maybe_function
            for child_scope_id in scope.children:
                scope = self.get_scope(child_scope_id)
                if not scope.namespace_name:
                    queue.append(child_scope_id)
            if scope.parent:
                parent = scope.parent
                if parent == scope_id:
                    self.compiler.panic(f'Scope {scope_id} is it\'s own parent!')
                queue.append(parent)
        return None

    def check_and_extract_weak_ptr(self, struct_id: StructId, args: List[TypeId]):
        weak_ptr_struct_id = self.find_struct_in_prelude('WeakPtr')

        if struct_id == weak_ptr_struct_id:
            if len(args) != 1:
                self.compiler.panic(f'Internal error: Generic type is WeakPtr, expected 1 type parameter, but got {len(args)} instead.')

    def type_name(self, type_id: TypeId) -> str:
        type = self.get_type(type_id)

        match type.variant:
            case 'Function':
                param_names: List[str] = []
                for param in type.params:
                    param_names.append(self.type_name(param))
                return_type = self.type_name(type.return_type_id)
                return f'function({", ".join(param_names)}) -> {return_type}'
            case 'Enum':
                return self.get_enum(type.id_).name
            case 'Struct':
                return self.get_struct(type.id_).name
            case 'GenericEnumInstance':
                return f'enum {self.get_enum(type.id_).name}<{", ".join([self.type_name(arg) for arg in type.args])}>'
            case 'GenericInstance':

                array_struct_id = self.find_struct_in_prelude("Array")
                dictionary_struct_id = self.find_struct_in_prelude("Dictionary")
                optional_struct_id = self.find_struct_in_prelude("Optional")
                range_struct_id = self.find_struct_in_prelude("Range")
                set_struct_id = self.find_struct_in_prelude("Set")
                tuple_struct_id = self.find_struct_in_prelude("Tuple")
                weak_ptr_struct_id = self.find_struct_in_prelude("WeakPtr")

                match type.id_:
                    case array_struct_id.id_:
                        return f'[{self.type_name(type.args[0])}]'
                    case dictionary_struct_id.id_:
                        return f'[{self.type_name(type.args[0])}:{self.type_name(type.args[1])}]'
                    case optional_struct_id.id_:
                        return f'{self.type_name(type.args[0])}?'
                    case range_struct_id.id_:
                        return f'{self.type_name(type.args[0])}..{self.type_name(type.args[0])}'
                    case set_struct_id.id_:
                        return f'{{{self.type_name(type.args[0])}}}'
                    case tuple_struct_id.id_:
                        return f'({", ".join([self.type_name(arg) for arg in type.args])})'
                    case weak_ptr_struct_id.id_:
                        return f'weak {self.type_name(type.args[0])}'
                    case _:
                        name = self.get_struct(type.id_).name
                        return f'{name}<{", ".join([self.type_name(arg) for arg in type.args])}>'
            case 'GenericResolvedType':
                name = self.get_struct(type.id_).name
                type_names = [self.type_name(type_id) * len(type.args)]
                return f'{name}<{", ".join(type_names)}>'
            case 'TypeVariable':
                return type.name
            case 'RawPtr':
                return f'raw {self.type_name(type_id)}'
            case 'Reference':
                return f'&{self.type_name(type_id)}'
            case 'MutableReference':
                return f'&mut {self.type_name(type_id)}'
            case _:
                return type.variant.lower()
