# Copyright (c) 2022-2022 Blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

from compiler.lexing.util import TextSpan


@dataclass
class ParsedGenericParameter:
    name: str
    span: TextSpan


@dataclass
class ParsedFunction:
    name: str
    name_span: TextSpan
    visibility: 'Visibility'
    params: List['ParsedParameter']
    generic_parameters: List[ParsedGenericParameter]
    block: 'ParsedBlock'
    return_type: 'ParsedType'
    return_type_span: TextSpan
    can_throw: bool
    type: 'FunctionType'
    linkage: 'FunctionLinkage'
    must_instantiate: bool
    is_comptime: bool


@dataclass
class ParsedVarDecl:
    name: str
    parsed_type: 'ParsedType'
    is_mutable: bool
    inlay_span: TextSpan | None # wtf is this?
    span: TextSpan


@dataclass
class ParsedMethod:
    parsed_function: ParsedFunction
    visibility: 'Visibility'


@dataclass
class ParsedVariable:
    name: str
    parsed_type: 'ParsedType'
    is_mutable: bool
    span: TextSpan


@dataclass
class ParsedCall:
    namespace: List[str]
    name: str
    args: List[Tuple[str, TextSpan, 'ParsedExpression']]
    type_args: List['ParsedType']


@dataclass
class ImportName:
    name: str
    span: TextSpan


@dataclass
class ParsedField:
    var_decl: ParsedVarDecl
    visibility: 'Visibility'


@dataclass
class ParsedModuleImport:
    module_name: ImportName
    alias_name: ImportName | None
    import_list: List[ImportName]

    def is_equivalent_to(self, other: 'ParsedModuleImport'):
        return self.module_name == other.module_name and self.has_same_alias_as(other) and self.has_same_import_semantics_as(other)

    def has_same_alias_as(self, other: 'ParsedModuleImport'):
        if self.alias_name is not None:
            return other.alias_name is not None and other.alias_name.name == self.alias_name.name
        else:
            return other.alias_name is None

    # In jakt, imports with an empty import list mean a namespaced import
    def has_same_import_semantics_as(self, other: 'ParsedModuleImport'):
        empty = len(self.import_list) == 0
        other_empty = len(other.import_list) == 0
        return empty == other_empty

    def merge_import_list(self, list_: List[ImportName]):
        name_set: set = set()
        for import_ in self.import_list:
            name_set.add(import_.name)
        for import_ in list_:
            if import_.name not in name_set:
                name_set.add(import_.name)
                self.import_list.append(import_)



@dataclass
class ParsedExternImport:
    is_c: bool
    assigned_namespace: 'ParsedNamespace'

    def get_path(self):
        return self.assigned_namespace.import_path_if_extern

    def get_name(self):
        return self.assigned_namespace.name

    def is_equivalent_to(self, other: 'ParsedExternImport'):
        return self.is_c and other.is_c and self.get_path() == other.get_path() and self.get_name() == other.get_name()


@dataclass
class ParsedNamespace:
    name: str | None
    name_span: TextSpan | None
    functions: List[ParsedFunction]
    records: List['ParsedRecord']
    namespaces: List['ParsedNamespace']
    module_imports: List[ParsedModuleImport]
    extern_imports: List[ParsedExternImport]
    import_path_if_extern: str | None

    def equivalent_to(self, other: 'ParsedNamespace'):
        return self.name == other.name and self.import_path_if_extern == other.import_path_if_extern

    def add_module_import(self, import_: ParsedModuleImport):
        for mod_import in self.module_imports:
            if mod_import.is_equivalent_to(import_):
                mod_import.merge_import_list(import_.import_list)
                return
        self.module_imports.append(import_)

    def add_extern_import(self, import_: ParsedExternImport):
        for extern_import in self.extern_imports:
            if extern_import.is_equivalent_to(import_):
                extern_import.assigned_namespace.merge_with(import_.assigned_namespace)
                return
        self.extern_imports.append(import_)

    def add_child_namespace(self, namespace: 'ParsedNamespace'):
        ...

    def merge_with(self, other: 'ParsedNamespace'):
        self.functions.extend(other.functions)
        self.records.extend(other.records)

        for mod_import in other.module_imports:
            self.add_module_import(mod_import)
        for extern_import in other.extern_imports:
            self.add_extern_import(extern_import)
        for child_namespace in other.namespaces:
            self.add_child_namespace(child_namespace)


@dataclass
class ValueEnumVariant:
    name: str
    span: TextSpan
    value: Union['ParsedExpression', None]


@dataclass
class SumEnumVariant:
    name: str
    span: TextSpan
    params: List[ParsedVarDecl] | None


@dataclass
class ParsedRecord:
    name: str
    name_span: TextSpan
    generic_parameters: List[ParsedGenericParameter]
    definition_linkage: 'DefinitionLinkage'
    methods: List[ParsedMethod]
    record_type: 'RecordType'


@dataclass
class ParsedMatchCase:
    patterns: List['ParsedMatchPattern']
    marker_span: TextSpan
    body: 'ParsedMatchBody'

    def __eq__(self, other: 'ParsedMatchCase'):
        if len(self.patterns) != len(other.patterns):
            return False
        for lhs_pattern, rhs_pattern in zip(self.patterns, other.patterns):
            if lhs_pattern != rhs_pattern:
                return False
        return self.body == other.body


@dataclass
class ParsedBlock:
    stmts: List['ParsedStatement']

    def find_yield_span(self) -> TextSpan | None:
        for stmt in self.stmts:
            if stmt.variant == 'Yield':
                return stmt.span
        return None

    def span(self, parser: 'Parser') -> TextSpan | None:
        start = None
        end = 0

        for stmt in self.stmts:
            stmt_span = stmt.span
            if start is not None:
                start = stmt.span
            end = stmt_span.end
        if start is not None:
            return parser.span(start, end)

    def __eq__(self, other: 'ParsedBlock'):
        if len(self.stmts) != len(other.stmts):
            return False
        for lhs_stmt, rhs_stmt in zip(self.stmts, other.stmts):
            if lhs_stmt != rhs_stmt:
                return False
        return True


@dataclass
class ParsedParameter:
    requires_label: bool
    variable: ParsedVariable
    default_argument: Union['ParsedExpression', None]
    span: TextSpan


@dataclass
class EnumVariantPatternArgument:
    name: str | None
    binding: str
    span: TextSpan
