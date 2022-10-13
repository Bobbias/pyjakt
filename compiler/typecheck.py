# Copyright (c) 2022-2022 blair 'Bobbias' Stacey
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause-Clear
from ctypes import c_long, c_byte, c_short, c_int, c_ulong, c_double
from pathlib import Path
from pprint import pprint, pformat
from typing import Dict, Tuple, Set, List

from compiler.compiler import Compiler
from compiler.error import CompilerError
from compiler.lexer import Lexer
from compiler.lexing.util import TextSpan, FileId, NumericConstant
from compiler.parsedtypes import ParsedNamespace, ParsedRecord, ParsedModuleImport, ParsedExternImport, ParsedFunction, \
    ParsedParameter, ParsedBlock, ParsedVarDecl, ParsedCall, ParsedMatchCase, EnumVariantPatternArgument
from compiler.parsing import Parser, Visibility, FunctionLinkage, FunctionType, ParsedExpression, ParsedType, \
    UnaryOperator, TypeCast, DefinitionLinkage, BinaryOperator, ParsedStatement, ParsedMatchBody, ParsedCapture
from compiler.types import TypeId, FunctionId, ModuleId, CheckedProgram, VarId, EnumId, StructId, ScopeId, \
    CheckedVariable, LoadedModule, Scope, Type, Module, CheckedExpression, unknown_type_id, CheckedEnum, builtin, \
    BuiltinType, CheckedBlock, BlockControlFlow, CheckedFunction, FunctionGenericParameter, CheckedParameter, \
    CheckedStruct, SafetyMode, CheckedEnumVariant, StructOrEnumId, never_type_id, void_type_id, CheckedStatement, \
    CheckedUnaryOperator, flip_signedness, CheckedNumericConstant, NumberConstant, CheckedEnumVariantBinding, \
    CheckedMatchBody, ResolvedNamespace, CheckedCall, CheckedMatchCase, CheckedTypeCast, CheckedCapture, \
    CheckedNamespace

builtin_types = [
        Type.Void(),
        Type.Bool(),
        Type.U8(),
        Type.U16(),
        Type.U32(),
        Type.U64(),
        Type.I8(),
        Type.I16(),
        Type.I32(),
        Type.I64(),
        Type.F32(),
        Type.F64(),
        Type.Usize(),
        Type.String(),
        Type.CChar(),
        Type.CInt(),
        Type.Unknown(),
        Type.Never()]


class Typechecker:
    compiler: Compiler
    program: CheckedProgram
    current_module_id: ModuleId
    current_struct_type_id: TypeId | None
    current_function_id: FunctionId | None
    inside_defer: bool
    checkidx: int
    ignore_errors: bool
    dump_type_hints: bool
    dump_try_hints: bool
    lambda_count: int

    def __init__(self, compiler: Compiler, program: CheckedProgram, current_module_id: ModuleId,
                 current_struct_type_id: TypeId | None, current_function_id: FunctionId | None,
                 inside_defer: bool, checkidx: int, ignore_errors: bool, dump_type_hints: bool,
                 dump_try_hints: bool, lambda_count: int):
        self.compiler = compiler
        self.program = program
        self.current_module_id = current_module_id
        self.current_struct_type_id = current_struct_type_id
        self.current_function_id = current_function_id
        self.inside_defer = inside_defer
        self.checkidx = checkidx
        self.ignore_errors = ignore_errors
        self.dump_type_hints = dump_type_hints
        self.dump_try_hints = dump_try_hints
        self.lambda_count = lambda_count

    def type_name(self, type_id: TypeId):
        return type_id.variant

    def dump_type_hint(self, type_id: TypeId, span: TextSpan):
        print(f"{{\"type\":\"hint\",\"file_id\":{span.file_id.id},\"position\":{span.end},\"typename\":\"{self.type_name(type_id)}\"}}")

    def dump_try_hint(self, span: TextSpan):
        print(f"{{\"type\":\"try\",\"file_id\":{span.file_id.id},\"position\":{span.start}}}")

    @classmethod
    def typecheck(cls, compiler: Compiler, parsed_namespace: ParsedNamespace) -> CheckedProgram:
        input_file = compiler.current_file

        if not input_file:
            compiler.panic('Trying to typecheck a non-existant file')

        placeholder_module_id = ModuleId(0)

        typechecker = Typechecker(
                compiler,
                CheckedProgram(compiler, [], {}),
                placeholder_module_id,
                TypeId.none(),
                None,
                False,
                0,
                False,
                compiler.dump_type_hints,
                compiler.dump_try_hints,
                0)

        typechecker.include_prelude()

        root_module_name = 'Root Module'
        root_module_id = typechecker.create_module(root_module_name, is_root=True)
        typechecker.current_module_id = root_module_id
        compiler.set_current_file(input_file)
        print(f'typecheck: setting loaded module {root_module_name}')
        typechecker.program.set_loaded_module(
                root_module_name,
                LoadedModule(root_module_id, input_file))
        PRELUDE_SCOPE_ID = typechecker.prelude_scope_id()
        root_scope_id = typechecker.create_scope(PRELUDE_SCOPE_ID, False, 'Root')
        print('typechecking root module')
        typechecker.typecheck_module(parsed_namespace, root_scope_id)

        return typechecker.program

    def get_function(self, id_: FunctionId):
        return self.program.get_function(id_)

    def get_variable(self, id_: VarId):
        return self.program.get_variable(id_)

    def get_type(self, id_: TypeId):
        return self.program.get_type(id_)

    def get_enum(self, id_: EnumId):
        return self.program.get_enum(id_)

    def get_struct(self, id_: StructId):
        return self.program.get_struct(id_)

    def get_scope(self, id_: ScopeId):
        return self.program.get_scope(id_)

    def find_var_in_scope(self, scope_id: ScopeId, var: str) -> CheckedVariable | None:
        return self.program.find_var_in_scope(scope_id, var)

    def get_root_path(self):
        file_id = self.program.get_loaded_module('Root Module').file_id
        return self.compiler.get_file_path(file_id)

    def prelude_scope_id(self) -> ScopeId:
        return self.program.prelude_scope_id()

    def root_scope_id(self) -> ScopeId:
        return ScopeId(ModuleId(1), 0)

    def current_module(self) -> Module:
        return self.program.get_module(self.current_module_id)

    def scope_can_access(self, accessor: ScopeId, accessee: ScopeId):
        if accessor == accessee:
            return True
        accessor_scope = self.get_scope(accessor)
        while accessor_scope.parent:
            parent = accessor_scope.parent
            if parent == accessee:
                return True
            accessor_scope = self.get_scope(parent)
        return False

    def error(self, message: str, span: TextSpan):
        if not self.ignore_errors:
            self.compiler.errors.append(CompilerError.Message(message, span))

    def error_with_hint(self, message: str, span: TextSpan, hint: str, hint_span: TextSpan):
        if not self.ignore_errors:
            self.compiler.errors.append(CompilerError.MessageWithHint(message, span, hint, hint_span))

    def is_integer(self, type_id: TypeId) -> bool:
        return self.program.is_integer(type_id)

    def is_floating(self, type_id: TypeId) -> bool:
        return self.program.is_floating(type_id)

    def is_numeric(self, type_id: TypeId) -> bool:
        return self.program.is_numeric(type_id)

    def create_scope(self, parent_scope_id: ScopeId | None, can_throw: bool, debug_name: str) -> ScopeId:
        if parent_scope_id:
            if parent_scope_id.module_id.id_ >= len(self.program.modules):
                self.compiler.panic(f'create_scope: parent_scope_id.module is invalid! '
                                    f'No module with id {parent_scope_id.module_id.id_}.')
            if parent_scope_id.id_ >= len(self.program.modules[parent_scope_id.module_id.id_].scopes):
                self.compiler.panic(f'create_scope: parent_scope_id.id_ is invalid! '
                                    f'Module {parent_scope_id.module_id.id_} does not have a scope'
                                    f' with id {parent_scope_id.id_}.')

        none_string: str | None = None

        scope = Scope(
                none_string,
                {},
                {},
                {},
                {},
                {},
                {},
                parent_scope_id,
                [],
                can_throw,
                None,
                debug_name)
        self.program.modules[self.current_module_id.id_].scopes.append(scope)

        return ScopeId(self.current_module_id, len(self.program.modules[self.current_module_id.id_].scopes) - 1)

    def create_module(self, name: str, is_root: bool) -> ModuleId:
        new_id = len(self.program.modules)
        module_id = ModuleId(new_id)
        module = Module(
                module_id,
                name,
                [],
                [],
                [],
                [],
                builtin_types,
                [],
                [],
                is_root)
        self.program.modules.append(module)
        return module_id

    # FIXME: get_prelude_contents() is a comptime function...
    #  Figure out if this has any implications for our implementation
    def get_prelude_contents(self):
        with Path('./runtime/prelude.jakt').resolve().open(mode='r') as file:
            return file.read()

    def include_prelude(self):
        module_name = '__prelude__'
        file_name = module_name  # in Jakt, this is a FilePath created using module_name resulting in a bogus path
        file_contents = self.get_prelude_contents()

        old_file_id = self.compiler.current_file
        old_file_contents = self.compiler.current_file_contents

        file_id = self.compiler.get_file_id_or_register(file_name)

        self.compiler.current_file = file_id
        self.compiler.current_file_contents = file_contents

        prelude_module_id = self.create_module(module_name, is_root=False)
        self.current_module_id = prelude_module_id
        print(f'include_prelude: setting loaded module: {prelude_module_id}')
        self.program.set_loaded_module(module_name, LoadedModule(prelude_module_id, file_id))
        print(f'current module: {self.current_module()}')

        prelude_scope_id = self.create_scope(None, can_throw=False, debug_name='prelude')
        tokens = Lexer(self.compiler).lex()

        if self.compiler.dump_lexer:
            for token in tokens:
                print(token)

        parsed_namespace = Parser.parse(self.compiler, tokens)

        if self.compiler.dump_parser:
            pprint(parsed_namespace)

        if self.program.modules:
            print('before typechecking parsed prelude, modules ', end='')
            print(pformat(self.program.modules, indent=4), end='')
            print('`')
        else:
            print('program.modules is empty')
        print(f'prelude module id: {self.current_module_id}')
        print(f'current module: {self.current_module()}')
        self.typecheck_module(parsed_namespace, prelude_scope_id)
        print('after typechecking prelude module')

        # Note: this action was deferred in the Jakt source, I'm not sure if that's a problem, here or not
        self.compiler.current_file = old_file_id
        self.compiler.current_file_contents = old_file_contents

    def lex_and_parse_file_contents(self, file_id: FileId):
        old_file_id = self.compiler.current_file

        if not self.compiler.set_current_file(file_id):
            return None

        tokens = Lexer(self.compiler).lex()

        if self.compiler.dump_lexer:
            for token in tokens:
                print(token)

        parsed_namespace = Parser.parse(self.compiler, tokens)

        if self.compiler.dump_parser:
            pprint(parsed_namespace)

        # Note: This was deferred in the Jakt source.
        self.compiler.set_current_file(old_file_id)

        return parsed_namespace

    def find_struct_in_prelude(self, name: str) -> StructId:
        return self.program.find_struct_in_prelude(name)

    def find_type_in_prelude(self, name: str):
        scope_id = self.prelude_scope_id()
        type_id = self.find_type_in_scope(scope_id, name)
        if type_id:
            return type_id
        self.compiler.panic(f'Internal error: {name} builtin definition not found')

    def try_to_promote_constant_expr_to_type(self, lhs_type: TypeId, checked_rhs: CheckedExpression, span: TextSpan):
        if not self.is_integer(lhs_type):
            return None

        rhs_constant = checked_rhs.to_number_constant(self.program)
        if not rhs_constant:
            return None

        result = rhs_constant.promote(lhs_type, self.program)
        if not result:
            type_ = self.get_type(lhs_type)
            self.error_with_hint('Integer promotion failed', span,
                                 f'Cannot fit value into range [{type_.min()}, {type_.max()}] '
                                 f'of type {self.type_name(lhs_type)}', span)
            return None
        return CheckedExpression.NumericConstant(result, span, lhs_type)

    def unify(self, lhs: TypeId, lhs_span: TextSpan, rhs: TypeId, rhs_span: TextSpan):
        if lhs.id_ != rhs.id_:
            self.error('Types incompatible', rhs_span)
            return None
        else:
            return lhs

    def unify_with_type(self, found_type: TypeId, expected_type: TypeId | None, span: TextSpan) -> TypeId:
        if not expected_type:
            return found_type
        if expected_type == unknown_type_id():
            return found_type
        generic_inferences: Dict[str, str] = {}
        if self.check_types_for_compat(expected_type, found_type, generic_inferences, span):
            return found_type
        return self.substitute_typevars_in_type(found_type, generic_inferences)

    def find_or_add_type_id(self, type_: Type) -> TypeId:
        for module in self.program.modules:
            for id_ in range(0, len(module.types)):
                if module.types[id_] == type_:
                    return TypeId(module.id_, id_)
        print(f'appending `{type_}` to module id `{self.current_module_id.id_}`')
        self.program.modules[self.current_module_id.id_].types.append(type_)
        return TypeId(self.current_module_id, len(self.current_module().types) - 1)

    def find_type_in_scope(self, scope_id: ScopeId, name: str) -> TypeId | None:
        current = scope_id

        while True:
            scope = self.get_scope(current)
            type_ = scope.types.get(name, None)
            if type_:
                return type_
            for child_id in scope.children:
                child_scope = self.get_scope(child_id)
                if not child_scope.namespace_name:
                    type_ = child_scope.types.get(name, None)
                    if type_:
                        return type_
            if scope.parent:
                current = scope.parent
            else:
                break

        return None

    def find_type_scope(self, scope_id: ScopeId, name: str) -> Tuple[TypeId, ScopeId] | None:
        current = scope_id
        print(f'find_type_scope: type: {name}, scope_id: {scope_id}')

        while True:
            scope = self.get_scope(current)
            print(f'find_type_scope: scope: {scope.debug_name}')
            type_ = scope.types.get(name, None)
            if type_:
                print(f'find_type_scope: found `{name}` in scope {scope.debug_name}')
                return type_, current
            print(f'find_type_scope: `{name}` not found in scope {scope.debug_name}, checking children')
            for child_id in scope.children:
                child_scope = self.get_scope(child_id)
                if not child_scope.namespace_name:
                    type_ = child_scope.types.get(name, None)
                    print(f'find_type_scope: listing type names in scope {scope.debug_name}')
                    for child_type_name, child_type_id in child_scope.types:
                        print(f'{child_type_name}: {child_type_id}')
                    if type_:
                        print(f'find_type_scope: found `{name}` as {type_} in scope {child_scope.debug_name}')
                        return type_, child_id
                    print(f'find_type_scope: `{name}` not found in current scope\'s children')
            if scope.parent:
                print(f'find_type_scope: `{name}` not found in current scope, checking parent scope')
                current = scope.parent
            else:
                print('find_type_scope: current scope has no parent, breaking loop')
                break
        print(f'find_type_scope: Cannot find `{name}` in any scope, returning None')
        return None

    def find_namespace_in_scope(self, scope_id: ScopeId, name: str) -> Tuple[ScopeId, bool] | None:
        return self.program.find_namespace_in_scope(scope_id, name)

    def add_struct_to_scope(self, scope_id: ScopeId, name: str, struct_id: StructId, span: TextSpan) -> bool:
        scope = self.get_scope(scope_id)
        maybe_struct_id = scope.structs.get(name, None)
        if maybe_struct_id:
            existing_struct_id = maybe_struct_id
            definition_span = self.get_struct(existing_struct_id).name_span

            self.error_with_hint(f'redefinition of struct/class {name}', span,
                                 f'struct/class {name} was first defined here', definition_span)
            return False
        scope.structs[name] = struct_id
        return True

    def add_enum_to_scope(self, scope_id: ScopeId, name: str, enum_id: EnumId, span: TextSpan) -> bool:
        scope = self.get_scope(scope_id)
        maybe_enum_id = scope.enums.get(name, None)
        if maybe_enum_id:
            existing_enum_id = maybe_enum_id
            definition_span = self.get_enum(existing_enum_id).name_span

            self.error_with_hint(f'redefinition of enum {name}', span,
                                 f'enum {name} was first defined here', definition_span)
            return False
        scope.enums[name] = enum_id
        return True

    def add_type_to_scope(self, scope_id: ScopeId, type_name: str, type_id: TypeId, span: TextSpan):
        scope = self.get_scope(scope_id)
        found_type_id = scope.types.get(type_name, None)
        if found_type_id:
            self.error(f'Redefinition of type `{type_name}`', span)
            return False
        scope.types[type_name] = type_id

    def add_function_to_scope(self, parent_scope_id: ScopeId, name: str, function_id: FunctionId, span: TextSpan):
        scope = self.get_scope(parent_scope_id)
        for existing_name, existing_function in scope.functions.items():
            if name == existing_name:
                function = self.get_function(existing_function)
                self.error_with_hint(f'Redefinition of function `{name}`', span,
                                     'previous definition here', function.name_span)
                return False
        scope.functions[name] = function_id
        return True

    def add_var_to_scope(self, scope_id: ScopeId, name: str, var_id: VarId, span: TextSpan):
        scope = self.get_scope(scope_id)
        for existing_name, existing_var in scope.vars.items():
            if name == existing_name:
                variable = self.get_variable(existing_var)
                self.error_with_hint(f'Redefinition of variable `{name}`', span,
                                     'previous definition here', variable.definition_span)
                return False
        scope.vars[name] = var_id


    def find_function_in_scope(self, parent_scope_id: ScopeId, function_name: str):
        return self.program.find_function_in_scope(parent_scope_id, function_name)

    def find_struct_in_scope(self, scope_id: ScopeId, name: str):
        return self.program.find_struct_in_scope(scope_id, name)

    def typecheck_module(self, parsed_namespace: ParsedNamespace, scope_id: ScopeId):
        if not self.current_module():
            self.compiler.panic(f'typecheck_module: Missing current_module! current_module_id: {self.current_module_id}')

        self.typecheck_namespace_imports(parsed_namespace, scope_id)
        self.typecheck_namespace_predecl(parsed_namespace, scope_id)
        self.typecheck_namespace_fields(parsed_namespace, scope_id)
        self.typecheck_namespace_constructors(parsed_namespace, scope_id)
        self.typecheck_namespace_function_predecl(parsed_namespace, scope_id)
        self.typecheck_namespace_declarations(parsed_namespace, scope_id)

    def typecheck_namespace_fields(self, parsed_namespace: ParsedNamespace, scope_id: ScopeId):
        children = self.get_scope(scope_id).children
        for i in range(0, len(parsed_namespace.namespaces)):
            child_namespace = parsed_namespace.namespaces[i]
            child_namespace_scope_id = children[i]
            self.typecheck_namespace_fields(child_namespace, child_namespace_scope_id)
        for record in parsed_namespace.records:
            if record.record_type.variant in ['Struct', 'Class']:
                struct_id = self.find_struct_in_scope(scope_id, record.name)
                if not struct_id:
                    self.compiler.panic('can\'t find previously added struct')
                self.typecheck_struct_fields(record, struct_id)

    def typecheck_struct_fields(self, record: ParsedRecord, struct_id: StructId):
        structure = self.get_struct(struct_id)

        checked_struct_scope_id = self.get_struct(struct_id).scope_id
        struct_type_id = self.find_or_add_type_id(Type.Struct(struct_id))
        self.current_struct_type_id = struct_type_id

        if record.record_type.variant not in ['Struct', 'Class']:
            self.compiler.panic('typecheck_struct_fields: cannot handle non-structs')
        parsed_fields = record.record_type.fields

        for unchecked_member in parsed_fields:
            print(f'typecheck_struct_fields: checking member: {unchecked_member}')
            parsed_var_decl = unchecked_member.var_decl
            print(f'typecheck_struct_fields: parsed_var_decl: {parsed_var_decl}')
            print(f'typecheck_struct_fields: typechecking typename `{parsed_var_decl.name}` in scope: {checked_struct_scope_id} with parsed type: {parsed_var_decl.parsed_type}')
            checked_member_type = self.typecheck_typename(parsed_var_decl.parsed_type, checked_struct_scope_id,
                                                          parsed_var_decl.name)

            print(f'typecheck_struct_fields: checked_member_type: {checked_member_type}')

            self.check_that_type_doesnt_contain_reference(checked_member_type, parsed_var_decl.parsed_type.span)

            module = self.current_module()
            var_id = module.add_variable(CheckedVariable(parsed_var_decl.name, checked_member_type, parsed_var_decl.is_mutable, parsed_var_decl.span, None, unchecked_member.visibility))
            structure.fields.append(var_id)

    def typecheck_module_import(self, import_: ParsedModuleImport, scope_id: ScopeId):
        imported_module_id = ModuleId(0)
        maybe_loaded_module = self.program.get_loaded_module(import_.module_name.name)
        if not maybe_loaded_module:
            maybe_file_name = self.compiler.search_for_path(import_.module_name.name)
            if maybe_file_name:
                file_name = maybe_file_name
            else:
                file_name = Path(f'{self.get_root_path()}/{import_.module_name.name}.jakt')

            file_id = self.compiler.get_file_id_or_register(file_name)

            parsed_namespace = self.lex_and_parse_file_contents(file_id)

            if not parsed_namespace:
                self.error(f'Module {import_.module_name.name} not found', import_.module_name.span)
                return

            original_current_module_id = self.current_module_id

            imported_module_id = self.create_module(import_.module_name.name, is_root=False)
            print(f'typecheck_module_import: setting loaded module {import_.module_name.name}')
            self.program.set_loaded_module(import_.module_name.name, LoadedModule(imported_module_id, file_id))
            self.current_module_id = imported_module_id

            imported_scope_id = self.create_scope(self.root_scope_id(), can_throw=False,
                                                  debug_name=f'module({import_.module_name.name})')
            print('typechecking imported module')
            self.typecheck_module(parsed_namespace, imported_scope_id)

            self.current_module_id = original_current_module_id
        else:
            imported_module_id = maybe_loaded_module.module_id

        current_module_imports = self.current_module().imports
        current_module_imports.append(imported_module_id)

        if len(import_.import_list) == 0:
            scope_imports = self.get_scope(scope_id).imports
            import_name = import_.module_name.name
            if import_.alias_name:
                import_name = import_.alias_name
            scope_imports[import_name] = imported_module_id
        else:
            import_scope_id = ScopeId(imported_module_id, 0)
            for imported_name in import_.import_list:
                maybe_function_id = self.find_function_in_scope(import_scope_id, imported_name.name)
                if maybe_function_id:
                    self.add_function_to_scope(scope_id, imported_name.name, maybe_function_id, imported_name.span)

                maybe_enum_id = self.program.find_enum_in_scope(import_scope_id, imported_name.name)
                if maybe_enum_id:
                    self.add_enum_to_scope(scope_id, imported_name.name, maybe_enum_id, imported_name.span)

                maybe_type_id = self.find_type_in_scope(import_scope_id, imported_name.name)
                if maybe_type_id:
                    self.add_type_to_scope(scope_id, imported_name.name, maybe_type_id, imported_name.span)

                maybe_struct_id = self.find_struct_in_scope(import_scope_id, imported_name.name)
                if maybe_struct_id:
                    self.add_struct_to_scope(scope_id, imported_name.name, maybe_struct_id, imported_name.span)

    def typecheck_extern_import(self, import_: ParsedExternImport, scope_id: ScopeId):
        for f in import_.assigned_namespace.functions:
            if f.linkage.variant != 'External':
                self.error('Expected all functions in an `import extern` to be external', f.name_span)

            if import_.is_c and len(f.generic_parameters) > 0:
                self.error_with_hint(f'imported function `{f.name}` is delcared to have C linkage, but is generic',
                                     f.name_span, 'this function may not be generic', f.name_span)

            if len(f.block.stmts) > 0:
                self.error('imported function is not allowed to have a body', f.name_span)

            for record in import_.assigned_namespace.records:
                if record.definition_linkage.variant != 'External':
                    self.error('Expected all record in an `import extern` to be external', record.name_span)
                if import_.is_c and len(record.generic_parameters) > 0:
                    self.error_with_hint(f'imported {record.record_type.variant} `{record.name}` '
                                         f'is delcared to have C linkage, but is generic', record.name_span,
                                         f'this {record.record_type.variant} may not be generic', record.name_span)

    def typecheck_namespace_imports(self, parsed_namespace: ParsedNamespace, scope_id: ScopeId):
        for module_import in parsed_namespace.module_imports:
            self.typecheck_module_import(module_import, scope_id)
        for extern_import in parsed_namespace.extern_imports:
            self.typecheck_extern_import(extern_import, scope_id)

    def typecheck_namespace_constructors(self, parsed_namespace: ParsedNamespace, scope_id: ScopeId):
        children = self.get_scope(scope_id).children
        for i in range(0, len(parsed_namespace.namespaces)):
            child_namespace = parsed_namespace.namespaces[i]
            child_namespace_scope_id = children[i]
            self.typecheck_namespace_constructors(child_namespace, child_namespace_scope_id)
        for record in parsed_namespace.records:
            if record.record_type.variant in ['Struct', 'Class']:
                struct_id = self.find_struct_in_scope(scope_id, record.name)
                if not struct_id:
                    self.compiler.panic('can\'t find previously added struct')
                self.typecheck_struct_constructor(record, struct_id, scope_id)
            elif record.record_type.variant in ['SumEnum', 'ValueEnum']:
                enum_id = self.program.find_enum_in_scope(scope_id, record.name)
                if not enum_id:
                    self.compiler.panic('can\'t find previously added enum')
                self.typecheck_enum_constructor(record, enum_id, scope_id)

    def typecheck_namespace_function_predecl(self, parsed_namespace: ParsedNamespace, scope_id: ScopeId):
        children = self.get_scope(scope_id).children
        for i in range(0, len(parsed_namespace.namespaces)):
            child_namespace = parsed_namespace.namespaces[i]
            child_namespace_scope_id = children[i]
            self.typecheck_namespace_function_predecl(child_namespace, child_namespace_scope_id)
        for fun in parsed_namespace.functions:
            self.typecheck_function_predecl(fun, scope_id, this_arg_type_id=None)

    def typecheck_namespace_predecl(self, parsed_namespace: ParsedNamespace, scope_id: ScopeId):
        module_struct_len = len(self.current_module().structures)
        module_enum_len = len(self.current_module().enums)

        # 1. Initialize structs
        struct_index = 0
        enum_index = 0
        for parsed_record in parsed_namespace.records:
            if parsed_record.record_type.variant in ['Struct', 'Class']:
                self.typecheck_struct_predecl_initial(parsed_record, struct_index, module_struct_len, scope_id)
                struct_index += 1
            elif parsed_record.record_type.variant in ['SumEnum', 'ValueEnum']:
                self.typecheck_enum_predecl_initial(parsed_record, enum_index, module_enum_len, scope_id)
                enum_index += 1

        # 2. Typecheck subnamespaces
        for namespace in parsed_namespace.namespaces:
            debug_name = f'namespace({namespace.name if namespace.name else "unnamed-namespace"})'
            namespace_scope_id = self.create_scope(scope_id, can_throw=False, debug_name=debug_name)
            child_scope = self.get_scope(namespace_scope_id)
            child_scope.namespace_name = namespace.name
            child_scope.import_path_if_extern = namespace.import_path_if_extern
            parent_scope = self.get_scope(scope_id)
            parent_scope.children.append(namespace_scope_id)
            self.typecheck_namespace_predecl(namespace, namespace_scope_id)

        # 3. Typecheck struct predeclaration
        struct_index = 0
        enum_index = 0
        for parsed_record in parsed_namespace.records:
            struct_id = StructId(self.current_module_id, struct_index + module_struct_len)
            if parsed_record.record_type.variant in ['Struct', 'Class']:
                self.typecheck_struct_predecl(parsed_record, struct_id, scope_id)
                struct_index += 1
            elif parsed_record.record_type.variant in ['SumEnum', 'ValueEnum']:
                enum_id = EnumId(self.current_module_id, enum_index + module_enum_len)
                self.typecheck_enum_predecl(parsed_record, enum_id, scope_id)
                enum_index += 1

    def typecheck_enum_predecl_initial(self, parsed_record: ParsedRecord, enum_index: int,
                                       module_enum_len: int, scope_id: ScopeId):
        module_id = self.current_module_id
        enum_id = EnumId(self.current_module_id, enum_index + module_enum_len)
        module = self.current_module()
        module.types.append(Type.Enum(enum_id))

        enum_type_id = TypeId(module_id, len(self.current_module().types) - 1)
        self.add_type_to_scope(scope_id, parsed_record.name, enum_type_id, parsed_record.name_span)

        is_boxed = False
        if parsed_record.record_type.variant == 'SumEnum':
            is_boxed = parsed_record.record_type.is_boxed

        # Placeholder entry
        module.enums.append(CheckedEnum(parsed_record.name, parsed_record.name_span, [], [],
                                        self.prelude_scope_id(), parsed_record.definition_linkage,
                                        parsed_record.record_type, enum_type_id, enum_type_id, is_boxed))

    def typecheck_enum_predecl(self, parsed_record: ParsedRecord, enum_id: EnumId, scope_id: ScopeId):
        enum_type_id = self.find_or_add_type_id(Type.Enum(enum_id))
        enum_scope_id = self.create_scope(scope_id, can_throw=False, debug_name=f'enum({parsed_record.name})')

        self.add_enum_to_scope(scope_id, parsed_record.name, enum_id, parsed_record.name_span)

        is_extern = parsed_record.definition_linkage.variant == 'External'

        if parsed_record.record_type.variant == 'ValueEnum':
            underlying_type_id = self.typecheck_typename(parsed_record.record_type.underlying_type, scope_id, name=None)
        else:
            underlying_type_id = builtin(BuiltinType.Void())

        if parsed_record.record_type.variant == 'SumEnum':
            is_boxed = parsed_record.record_type.is_boxed
        else:
            is_boxed = False

        module = self.current_module()
        module.enums[enum_id.id_] = CheckedEnum(parsed_record.name, parsed_record.name_span, [], [],
                                                enum_scope_id, parsed_record.definition_linkage,
                                                parsed_record.record_type, underlying_type_id,
                                                enum_type_id, is_boxed)

        generic_parameters = module.enums[enum_id.id_].generic_parameters
        for gen_parameter in parsed_record.generic_parameters:
            module.types.append(Type.TypeVariable(gen_parameter.name))

            parameter_type_id = TypeId(self.current_module_id, len(self.current_module().types) - 1)

            generic_parameters.append(parameter_type_id)

            self.add_type_to_scope(enum_scope_id, gen_parameter.name, parameter_type_id, gen_parameter.span)

        for method in parsed_record.methods:
            func = method.parsed_function
            method_scope_id = self.create_scope(enum_scope_id, func.can_throw,
                                                f'method({parsed_record.name}::{func.name})')
            block_scope_id = self.create_scope(method_scope_id, func.can_throw,
                                               f'method-block({parsed_record.name}::{func.name})')
            if len(parsed_record.generic_parameters) > 0 or len(func.generic_parameters) > 0:
                is_generic = True
            else:
                is_generic = False
            checked_function = CheckedFunction(
                    func.name, func.name_span, method.visibility, unknown_type_id(), None, [], [],
                    CheckedBlock(
                            [], block_scope_id,BlockControlFlow.MayReturn(), TypeId.none()),
                    func.can_throw, func.type, func.linkage, method_scope_id, not is_generic or is_extern,
                    func, func.is_comptime)
            module.functions.append(checked_function)

            function_id = FunctionId(self.current_module_id, len(self.current_module().functions) - 1)
            generic_parameters = []

            for generic_parameter in func.generic_parameters:
                module.types.append(Type.TypeVariable(generic_parameter.name))
                type_var_type_id = TypeId(self.current_module_id, len(self.current_module().types) - 1)
                generic_parameters.append(FunctionGenericParameter.Parameter(type_var_type_id))
                if not func.must_instantiate:
                    self.add_type_to_scope(method_scope_id, generic_parameter.name,
                                           type_var_type_id, generic_parameter.span)

            checked_function.generic_params = generic_parameters

            for param in func.params:
                if param.variable.name == 'this':
                    checked_variable = CheckedVariable(param.variable.name, enum_type_id,
                                                       param.variable.is_mutable, param.variable.span,
                                                       None, Visibility.Public())
                    checked_function.params.append(CheckedParameter(param.requires_label, checked_variable, None))
                else:
                    param_type = self.typecheck_typename(param.variable.parsed_type, method_scope_id,
                                                         param.variable.name)
                    checked_variable = CheckedVariable(param.variable.name, param_type, param.variable.is_mutable,
                                                       param.variable.span, param.variable.parsed_type.span,
                                                       Visibility.Public())
                    checked_function.params.append(CheckedParameter(param.requires_label, checked_variable, None))

            self.add_function_to_scope(enum_scope_id, func.name, function_id, parsed_record.name_span)

            function_return_type_id = self.typecheck_typename(func.return_type, method_scope_id, None)
            checked_function.return_type_id = function_return_type_id

    def typecheck_struct_constructor(self, parsed_record: ParsedRecord, struct_id: StructId, scope_id: ScopeId):
        struct_type_id = self.find_or_add_type_id(Type.Struct(struct_id))
        self.current_struct_type_id = struct_type_id

        struct = self.get_struct(struct_id)

        constructor_id = self.find_function_in_scope(struct.scope_id, parsed_record.name)
        if constructor_id:
            if parsed_record.record_type.variant == 'Class' and parsed_record.definition_linkage.variant == 'External':
                # the parser always sets the linkage type of an extern class'
                # constructor to External, but we actually want to call the
                # class' ::create function, just like we do with a
                # ImplicitConstructor class.
                func = self.get_function(constructor_id)
                func.linkage = FunctionLinkage.External()
        elif parsed_record.definition_linkage.variant != 'External':
            # no constructor found, so let's make one
            constructor_can_throw = parsed_record.record_type.variant == 'Class'
            function_scope_id = self.create_scope(struct.scope_id, constructor_can_throw,
                                                  f'generated-constructor({parsed_record.name})')
            block_scope_id = self.create_scope(function_scope_id, constructor_can_throw,
                                               f'generated-constructor-block({parsed_record.name})')

            checked_constructor = CheckedFunction(parsed_record.name, parsed_record.name_span, Visibility.Public(),
                                                  struct_type_id, None, [], [],
                                                  CheckedBlock(
                                                          [], block_scope_id, BlockControlFlow.MayReturn(),
                                                          TypeId.none()),
                                                  constructor_can_throw, FunctionType.ImplicitConstructor(),
                                                  FunctionLinkage.Internal(), function_scope_id, True, None, False)

            module = self.current_module()
            module.functions.append(checked_constructor)

            func = module.functions[-1]
            for field_id in self.get_struct(struct_id).fields:
                field = self.get_variable(field_id)
                func.params.append(CheckedParameter(True, field, None))
            self.add_function_to_scope(struct.scope_id, parsed_record.name,
                                       FunctionId(self.current_module_id, len(self.current_module().functions) - 1),
                                       parsed_record.name_span)

        self.current_struct_type_id = None

    def typecheck_struct_predecl(self, parsed_record: ParsedRecord, struct_id: StructId, scope_id: ScopeId):
        struct_type_id = self.find_or_add_type_id(Type.Struct(struct_id))
        self.current_struct_type_id = struct_type_id

        struct_scope_id = self.create_scope(scope_id, False, f'struct({parsed_record.name})')

        self.add_struct_to_scope(scope_id, parsed_record.name, struct_id, parsed_record.name_span)

        is_extern = True if parsed_record.definition_linkage.variant == 'External' else False

        module = self.current_module()
        module.structures[struct_id.id_] = CheckedStruct(parsed_record.name, parsed_record.name_span, [], [],
                                                         struct_scope_id, parsed_record.definition_linkage,
                                                         parsed_record.record_type, struct_type_id)

        generic_parameters = module.structures[struct_id.id_].generic_parameters
        for gen_parameter in parsed_record.generic_parameters:
            module.types.append(Type.TypeVariable(gen_parameter.name))
            parameter_type_id = TypeId(self.current_module_id, len(self.current_module().types) - 1)
            generic_parameters.append(parameter_type_id)
            self.add_type_to_scope(struct_scope_id, gen_parameter.name, parameter_type_id, gen_parameter.span)

        for method in parsed_record.methods:
            func = method.parsed_function

            method_scope_id = self.create_scope(struct_scope_id, func.can_throw,
                                                f'method({parsed_record.name}::{func.name})')
            block_scope_id = self.create_scope(method_scope_id, func.can_throw,
                                               f'method-block({parsed_record.name}::{func.name})')
            if len(parsed_record.generic_parameters) > 0 or len(func.generic_parameters) > 0:
                is_generic = True
            else:
                is_generic = False
            checked_function = CheckedFunction(func.name, func.name_span, method.visibility, unknown_type_id(),
                                               func.return_type_span, [], [],
                                               CheckedBlock([], block_scope_id, BlockControlFlow.MayReturn(),
                                                            TypeId.none()),
                                               func.can_throw, func.type, func.linkage, method_scope_id,
                                               not is_generic or is_extern, method.parsed_function,
                                               method.parsed_function.is_comptime)
            module.functions.append(checked_function)
            function_id = FunctionId(self.current_module_id, len(self.current_module().functions) - 1)
            previous_index = self.current_function_id
            check_scope = None
            if is_generic:
                check_scope = self.create_scope(method_scope_id, func.can_throw,
                                                f'method-checking({parsed_record.name}::{func.name})')

            for gen_parameter in func.generic_parameters:
                module.types.append(Type.TypeVariable(gen_parameter.name))
                type_var_type_id = TypeId(self.current_module_id, len(self.current_module().types) - 1)
                checked_function.generic_params.append(FunctionGenericParameter.Parameter(type_var_type_id))
                self.add_type_to_scope(method_scope_id, gen_parameter.name, type_var_type_id, gen_parameter.span)

            for param in func.params:
                if param.variable.name == 'this':
                    checked_variable = CheckedVariable(param.variable.name, struct_type_id, param.variable.is_mutable,
                                                       param.variable.span, None, Visibility.Public())

                    checked_function.params.append(CheckedParameter(param.requires_label, checked_variable, None))
                    if check_scope:
                        var_id = module.add_variable(checked_variable)
                        self.add_var_to_scope(check_scope, param.variable.name, var_id, param.variable.span)
                else:
                    param_type = self.typecheck_typename(param.variable.parsed_type, method_scope_id,
                                                         param.variable.name)

                    checked_variable = CheckedVariable(param.variable.name, param_type, param.variable.is_mutable,
                                                       param.variable.span, param.variable.parsed_type.span,
                                                       Visibility.Public())

                    checked_default_value = None
                    if param.default_argument:
                        checked_default_value_expr = self.typecheck_expression(param.default_argument, scope_id,
                                                                               SafetyMode.Safe(), param_type)
                        if checked_default_value_expr.variant == 'OptionalNone':
                            expr_span = checked_default_value_expr.span
                            checked_default_value_expr = CheckedExpression.OptionalNone(expr_span, param_type)

                        default_value_type_id = checked_default_value_expr.type()
                        checked_default_value = checked_default_value_expr
                        if default_value_type_id != param_type:
                            checked_default_value = None
                            self.error(f'Type mismatch: expected `{self.type_name(param_type)}` got '
                                       f'`{self.type_name(default_value_type_id)}`', param.span)
                    checked_function.params.append(CheckedParameter(param.requires_label, checked_variable, checked_default_value))
                    if check_scope:
                        var_id = module.add_variable(checked_variable)
                        self.add_var_to_scope(check_scope, param.variable.name, var_id, param.variable.span)
            self.add_function_to_scope(struct_scope_id, func.name,
                                       FunctionId(self.current_module_id, len(self.current_module().functions) - 1),
                                       parsed_record.name_span)
            function_return_type_id = self.typecheck_typename(func.return_type, method_scope_id, None)
            checked_function.return_type_id = function_return_type_id

            if is_generic:
                if not check_scope:
                    self.compiler.panic('Generic method with generic parameters must have a check scope')

                old_ignore_errors = self.ignore_errors
                self.ignore_errors = True
                block = self.typecheck_block(func.block, check_scope, SafetyMode.Safe())
                self.ignore_errors = old_ignore_errors

                return_type_id = builtin(BuiltinType.Void())
                if function_return_type_id == unknown_type_id():
                    if len(block.statements) > 0:
                        last_statement = block.statements[-1]
                        val = last_statement.val
                        if last_statement.variant == 'Return' and last_statement.val:
                            return_type_id = self.resolve_type_var(val.type(), method_scope_id)
                else:
                    return_type_id = self.resolve_type_var(function_return_type_id, scope_id)
                checked_function.block = block
                checked_function.return_type_id = return_type_id
            module.functions[function_id.id_] = checked_function
            self.current_function_id = previous_index

        module.structures[struct_id.id_].generic_parameters = generic_parameters
        self.current_struct_type_id = None

    def typecheck_struct_predecl_initial(self, parsed_record: ParsedRecord, struct_index: int,
                                         module_struct_len: int, scope_id: ScopeId):
        module_id = self.current_module_id
        struct_id = EnumId(self.current_module_id, struct_index + module_struct_len)
        module = self.current_module()
        module.types.append(Type.Enum(struct_id))

        struct_type_id = TypeId(module_id, len(self.current_module().types) - 1)
        self.add_type_to_scope(scope_id, parsed_record.name, struct_type_id, parsed_record.name_span)

        # Placeholder entry
        module.structures.append(
                CheckedStruct(parsed_record.name, parsed_record.name_span, [], [], self.prelude_scope_id(),
                              parsed_record.definition_linkage, parsed_record.record_type, struct_type_id))

    def typecheck_namespace_declarations(self, parsed_namespace: ParsedNamespace, scope_id: ScopeId):
        children = self.get_scope(scope_id).children
        for i in range(0, len(parsed_namespace.namespaces)):
            child_namespace = parsed_namespace.namespaces[i]
            child_namespace_scope_id = children[i]
            self.typecheck_namespace_declarations(child_namespace, child_namespace_scope_id)

        for record in parsed_namespace.records:
            if record.record_type.variant in ['Struct', 'Class']:
                struct_id = self.find_struct_in_scope(scope_id, record.name)
                if not struct_id:
                    self.compiler.panic('can\'t find previously added struct')
                self.typecheck_struct(record, struct_id, scope_id)
            elif record.record_type.variant in ['SumEnum', 'ValueEnum']:
                enum_id = self.program.find_enum_in_scope(scope_id, record.name)
                if not enum_id:
                    self.compiler.panic('can\'t find previously added enum')
                self.typecheck_enum(record, enum_id, scope_id)

        for fun in parsed_namespace.functions:
            self.current_function_id = self.find_function_in_scope(scope_id, fun.name)
            self.typecheck_function(fun, scope_id)
            self.current_function_id = None

    def typecheck_enum_constructor(self, record: ParsedRecord, enum_id: EnumId, parent_scope_id: ScopeId):
        next_constant_value = 0
        seen_names = set()

        enum_ = self.get_enum(enum_id)

        if record.record_type.variant == 'ValueEnum':
            underlying_type = record.record_type.underlying_type
            variants = record.record_type.variants
            underlying_type_id = self.typecheck_typename(underlying_type, parent_scope_id, None)
            module = self.current_module()
            for variant in variants:
                if variant.name in seen_names:
                    self.error(f'Enum variant `{variant.name}` is defined more than once', variant.scope)
                else:
                    seen_names.add(variant.name)

                    if variant.value:
                        value_expression = self.cast_to_underlying(variant.value, parent_scope_id, underlying_type)
                        number_constant = value_expression.to_number_constant(self.program)
                        if number_constant:
                            if number_constant.variant == 'Floating':
                                # todo: implement floats
                                next_constant_value = 0
                            else:
                                next_constant_value = number_constant.val + 1
                        else:
                            self.error(f'Enum variant `{variant.name}` in enum `{enum_.name}` '
                                       f'has non-constant value: {value_expression}', variant.span)
                        expr = value_expression
                    else:
                        expr = self.cast_to_underlying(
                                ParsedExpression.NumericConstant(NumericConstant.U64(next_constant_value), variant.span),
                                parent_scope_id,
                                underlying_type)
                        next_constant_value += 1
                    enum_.variants.append(CheckedEnumVariant.WithValue(enum_id, variant.name, expr, variant.span))
                    var_id = module.add_variable(
                            CheckedVariable(variant.name, enum_.type_id, False,
                                            variant.span, None, Visibility.Public()))
                    self.add_var_to_scope(enum_.scope_id, variant.name, var_id, variant.span)
        elif record.record_type.variant == 'SumEnum':
            module = self.current_module()
            is_boxed = record.record_type.is_boxed
            for variant in record.record_type.variants:
                if variant.name in seen_names:
                    self.error(f'Enum variant `{variant.name}` is defined more than once', variant.span)
                    continue
                seen_names.add(variant.name)
                is_structlike = variant.params and len(variant.params) > 0 and variant.params[0].name != ''
                is_typed = variant.params and len(variant.params) == 1 and variant.params[0].name == ''
                if is_structlike:
                    seen_fields: Set[str] = set()
                    fields: List[VarId] = []
                    params: List[CheckedParameter] = []
                    for param in variant.params:
                        if param.name in seen_fields:
                            self.error(f'Enum variant `{variant.name}` has a member named '
                                       f'`{param.name}` more than once', variant.span)
                            continue
                        seen_fields.add(param.name)
                        type_id = self.typecheck_typename(param.parsed_type, enum_.scope_id, param.name)
                        checked_var = CheckedVariable(param.name, type_id, param.is_mutable, param.span,
                                                          None, Visibility.Public())
                        params.append(CheckedParameter(True, checked_var, None))

                        if self.dump_type_hints:
                            self.dump_type_hint(type_id, param.span)
                        var_id = module.add_variable(checked_var)
                        fields.append(var_id)
                    enum_.variants.append(
                            CheckedEnumVariant.StructLike(enum_id, variant.name, fields, variant.span))
                    maybe_enum_variant_constructor = self.find_function_in_scope(enum_.scope_id, variant.name)
                    if not maybe_enum_variant_constructor:
                        can_function_throw = is_boxed
                        function_scope_id = self.create_scope(parent_scope_id, can_function_throw,
                                                              f'enum-variant-constructor({enum_.name}::{variant.name})')
                        block_scope_id = self.create_scope(function_scope_id,
                                                           can_function_throw,
                                                           f'enum-variant-constructor-block({enum_.name}::{variant.name})')
                        checked_function = CheckedFunction(variant.name, variant.span, Visibility.Public(),
                                                           self.find_or_add_type_id(Type.Enum(enum_id)), None, params,
                                                           [], CheckedBlock(
                                                                   [], block_scope_id, BlockControlFlow.MayReturn(),
                                                                   TypeId.none()),
                                                           can_function_throw, FunctionType.ImplicitEnumConstructor(),
                                                           FunctionLinkage.Internal(), function_scope_id, True,
                                                           None, False)
                        function_id = module.add_function(checked_function)
                        self.add_function_to_scope(enum_.scope_id, variant.name, function_id, variant.span)
                elif is_typed:
                    param = variant.params[0]
                    type_id = self.typecheck_typename(param.parsed_type, enum_.scope_id, param.name)
                    enum_.variants.append(CheckedEnumVariant.Typed(enum_id, variant.name, type_id, variant.span))
                    maybe_enum_variant_constructor = self.find_function_in_scope(enum_.scope_id, variant.name)
                    if not maybe_enum_variant_constructor:
                        can_function_throw = is_boxed
                        function_scope_id = self.create_scope(parent_scope_id, can_function_throw,
                                                              f'enum-variant-constructor({enum_.name}::{variant.name})')
                        block_scope_id = self.create_scope(function_scope_id,
                                                           can_function_throw,
                                                           f'enum-variant-constructor-block({enum_.name}::{variant.name})')
                        variable = CheckedVariable('value', type_id, False, param.span, None, Visibility.Public())
                        checked_function = CheckedFunction(variant.name, variant.span, Visibility.Public(),
                                                           self.find_or_add_type_id(Type.Enum(enum_id)), None,
                                                           [CheckedParameter(False, variable, None)], [],
                                                           CheckedBlock(
                                                                   [], block_scope_id, BlockControlFlow.AlwaysReturns(),
                                                                   TypeId.none()),
                                                           can_function_throw, FunctionType.ImplicitEnumConstructor(),
                                                           FunctionLinkage.Internal(), function_scope_id,
                                                           True, None, False)
                        function_id = module.add_function(checked_function)
                        self.add_function_to_scope(enum_.scope_id, variant.name, function_id, variant.span)
                else:
                    enum_.variants.append(CheckedEnumVariant.Untyped(enum_id, variant.name, variant.span))
                    maybe_enum_variant_constructor = self.find_function_in_scope(enum_.scope_id, variant.name)
                    if not maybe_enum_variant_constructor:
                        can_function_throw = is_boxed
                        function_scope_id = self.create_scope(parent_scope_id, can_function_throw,
                                                              f'enum-variant-constructor({enum_.name}::{variant.name})')
                        block_scope_id = self.create_scope(function_scope_id,
                                                           can_function_throw,
                                                           f'enum-variant-constructor-block({enum_.name}::{variant.name})')
                        checked_function = CheckedFunction(variant.name, variant.span, Visibility.Public(),
                                                           self.find_or_add_type_id(Type.Enum(enum_id)), None,
                                                           [], [], CheckedBlock(
                                                                   [], block_scope_id, BlockControlFlow.AlwaysReturns(),
                                                                   TypeId.none()),
                                                           can_function_throw, FunctionType.ImplicitEnumConstructor(),
                                                           FunctionLinkage.Internal(), function_scope_id,
                                                           True, None, False)
                        function_id = module.add_function(checked_function)
                        self.add_function_to_scope(enum_.scope_id, variant.name, function_id, variant.span)

    def typecheck_enum(self, record: ParsedRecord, enum_id: EnumId, parent_scope_id: ScopeId):
        for method in record.methods:
            self.typecheck_method(method.parsed_function, StructOrEnumId.Enum(enum_id))

    def cast_to_underlying(self, expr: ParsedExpression, scope_id: ScopeId, parsed_type: ParsedType):
        cast_expression = ParsedExpression.UnaryOp(expr, UnaryOperator.TypeCast(TypeCast.Infallible(parsed_type)),
                                                   expr.span)
        return self.typecheck_expression(cast_expression, scope_id, SafetyMode.Safe(), None)

    def typecheck_struct(self, record: ParsedRecord, struct_id: StructId, parent_scope_id: ScopeId):
        struct_type_id = self.find_or_add_type_id(Type.Struct(struct_id))
        self.current_struct_type_id = struct_type_id
        for method in record.methods:
            self.typecheck_method(method.parsed_function, StructOrEnumId.Struct(struct_id))
        self.current_struct_type_id = None

    def typecheck_method(self, func: ParsedFunction, parent_id: StructOrEnumId):
        parent_generic_parameters: List[TypeId] = []
        scope_id = self.prelude_scope_id()
        definition_linkage = DefinitionLinkage.Internal()

        if parent_id.variant == 'Struct':
            structure = self.get_struct(parent_id.struct_id)
            parent_generic_parameters = structure.generic_parameters
            scope_id = structure.scope_id
            definition_linkage = structure.definition_linkage
        elif parent_id.variant == 'Enum':
            enum_ = self.get_enum(parent_id.enum_id)
            parent_generic_parameters = enum_.generic_parameters
            definition_linkage = enum_.definition_linkage
            scope_id = enum_.scope_id

        if len(func.generic_parameters) > 0 or len(parent_generic_parameters) > 0 and not func.must_instantiate:
            return

        structure_scope_id = scope_id
        structure_linkage = definition_linkage

        method_id = self.find_function_in_scope(structure_scope_id, func.name)
        if not method_id:
            self.compiler.panic('typecheck_function: We just pushed the checked function, but it\'s not present')

        checked_function = self.get_function(method_id)
        function_scope_id = checked_function.function_scope_id

        module = self.current_module()
        for param in checked_function.params:
            variable = param.variable
            var_id = module.add_variable(variable)
            self.add_var_to_scope(function_scope_id, variable.name, variable.definition_span)

        # set current function index before a block type check so that method return type
        # is checked against it's implementation
        self.current_function_id = method_id

        VOID_TYPE_ID = builtin(BuiltinType.Void())

        block = self.typecheck_block(func.block, function_scope_id, SafetyMode.Safe())
        function_return_type_id = self.typecheck_typename(func.return_type, function_scope_id, None)

        return_type_id = function_return_type_id
        if function_return_type_id == unknown_type_id() and len(block.statements) > 0:
            # if the return type is unknown, and the function starts with a return statement
            # we infer the return type from it's expression
            first_statement = block.statements[0]
            if first_statement.variant == 'Return' and first_statement.val:
                return_type_id = first_statement.val.type()
            else:
                return_type_id = VOID_TYPE_ID
        elif function_return_type_id == unknown_type_id():
            return_type_id = VOID_TYPE_ID

        if structure_linkage.variant != 'External' and return_type_id != VOID_TYPE_ID\
            and not block.control_flow.always_transfers_control():
            if return_type_id == never_type_id() and block.control_flow.never_returns():
                self.error('Control reaches end of never-returning function', func.name_span)
            elif not block.control_flow.never_returns():
                self.error('Control reaches end of non-void function', func.name_span)

        checked_function.block = block
        checked_function.return_type_id = return_type_id

    def typecheck_parameter(self, parameter: ParsedParameter, scope_id: ScopeId, first: bool,
                            this_arg_type_id: TypeId | None, check_scope: ScopeId | None):
        type_id = self.typecheck_typename(parameter.variable.parsed_type, scope_id, parameter.variable.name)

        if first and parameter.variable.name == 'this':
            if this_arg_type_id:
                type_id = this_arg_type_id

        variable = CheckedVariable(parameter.variable.name, type_id, parameter.variable.is_mutable,
                                   parameter.variable.span, None, Visibility.Public())

        checked_default_value: CheckedExpression | None = None
        if parameter.default_argument:
            # todo: currently this assumes that things are safe, check jakt again later to see if they
            #  implemented this or if we should attempt implementing this ourself
            checked_default_value_expr = self.typecheck_expression(parameter.default_argument, scope_id,
                                                                   SafetyMode.Safe(), type_id)
            if checked_default_value_expr.variant == 'OptionalNone':
                # todo: this is awkward, I think we can just set type_id. I believe in jakt these enums are immutable
                checked_default_value_expr = CheckedExpression.OptionalNone(checked_default_value_expr.span, type_id)

            default_value_type_id = checked_default_value_expr.type()
            checked_default_value = checked_default_value_expr
            if default_value_type_id != type_id:
                checked_default_value = None
                self.error(f'Type mismatch: expected `{self.type_name(type_id)}`,'
                           f' but got `{self.type_name(default_value_type_id)}`', parameter.span)

        checked_parameter = CheckedParameter(parameter.requires_label, variable, checked_default_value)

        if check_scope:
            module = self.current_module()
            var_id = module.add_variable(variable)
            self.add_var_to_scope(check_scope, parameter.variable.name, var_id, parameter.variable.span)

        return checked_parameter

    def typecheck_function_predecl(self, parsed_function: ParsedFunction, parent_scope_id: ScopeId,
                                   this_arg_type_id: TypeId | None):
        function_scope_id = self.create_scope(parent_scope_id, parsed_function.can_throw,
                                              f'function({parsed_function.name})')
        scope_debug_name = f'function-block({parsed_function.name})'
        block_scope_id = self.create_scope(function_scope_id, parsed_function.can_throw,
                                           scope_debug_name)
        module_id = self.current_module_id.id_

        is_generic_function = len(parsed_function.generic_parameters) > 0
        if this_arg_type_id:
            if self.get_type(this_arg_type_id).variant == 'GenericInstance':
                is_generic = True
            else:
                is_generic = is_generic_function
        else:
            is_generic = is_generic_function

        checked_function = CheckedFunction(
                parsed_function.name, parsed_function.name_span, parsed_function.visibility, unknown_type_id(),
                parsed_function.return_type_span, [], [],
                CheckedBlock([], block_scope_id, BlockControlFlow.MayReturn(), TypeId.none()),
                parsed_function.can_throw, FunctionType.Normal(), parsed_function.linkage, function_scope_id,
                not is_generic, parsed_function, parsed_function.is_comptime)
        # todo: We can't return a `mut Foo` from a function right now, but assigning anything to a `mut` variable
        #  makes it mutable, AKA, working around one bug with another bug. :^)
        #  We should check if this has been fixed later
        module = self.current_module()
        function_id = module.add_function(checked_function)
        checked_function_scope_id = checked_function.function_scope_id

        external_linkage = parsed_function.linkage.variant == 'External'

        if is_generic:
            check_scope = self.create_scope(parent_scope_id, parsed_function.can_throw, scope_debug_name)
        else:
            check_scope = None

        # check generic parameters
        for generic_parameter in parsed_function.generic_parameters:
            module.types.append(Type.TypeVariable(generic_parameter.name))
            type_var_type_id = TypeId(module.id_, len(module.types) - 1)
            checked_function.generic_params.append(FunctionGenericParameter.Parameter(type_var_type_id))

            if not parsed_function.must_instantiate or external_linkage:
                self.add_type_to_scope(checked_function_scope_id, generic_parameter.name, type_var_type_id,
                                       generic_parameter.span)
                if check_scope:
                    self.add_type_to_scope(check_scope, generic_parameter.name, type_var_type_id, generic_parameter.span)

        # check parameters
        first = True
        module = self.current_module()
        for parameter in parsed_function.params:
            checked_function.params.append(self.typecheck_parameter(parameter, checked_function_scope_id, first, this_arg_type_id, check_scope))
            first = False

        # check return type
        function_return_type_id = self.typecheck_typename(parsed_function.return_type, checked_function_scope_id, None)
        checked_function.return_type_id = function_return_type_id

        self.check_that_type_doesnt_contain_reference(function_return_type_id, parsed_function.return_type_span)

        if len(parsed_function.generic_parameters) > 0:
            old_ignore_errors = self.ignore_errors
            self.ignore_errors = True
            block = self.typecheck_block(parsed_function.block, check_scope, SafetyMode.Safe())
            self.ignore_errors = old_ignore_errors

            if function_return_type_id == unknown_type_id():
                if block.statements:
                    last_statement = block.statements[-1]
                    if last_statement.variant == 'Return':
                        if last_statement.val:
                            return_type_id = last_statement.val.type()
                        else:
                            return_type_id = void_type_id()
                    else:
                        return_type_id = void_type_id()
                else:
                    return_type_id = unknown_type_id()
            else:
                return_type_id = self.resolve_type_var(function_return_type_id, parent_scope_id)

            checked_function.block = block
            checked_function.return_type_id = return_type_id

        self.add_function_to_scope(parent_scope_id, parsed_function.name, function_id, parsed_function.name_span)

    def check_that_type_doesnt_contain_reference(self, type_id: TypeId, span: TextSpan):
        type_ = self.get_type(type_id)

        # FIXME: Check for any type that contains a reference as a generic parameter, etc.
        if type_.variant in ['Reference', 'MutableReference']:
            contains_reference = True
        else:
            contains_reference = False

        if contains_reference:
            self.error(f'Reference type `{self.type_name(type_id)}` not usable in this context', span)

    def typecheck_and_specialize_generic_function(self, function_id: FunctionId, generic_arguments: List[TypeId],
                                                  parent_scope_id: ScopeId, this_type_id: TypeId | None,
                                                  generic_substitutions: Dict[str, str]):
        checked_function = self.get_function(function_id)
        module = self.current_module()

        function_id = module.next_function_id()
        if not checked_function.parsed_function:
            return
        parsed_function = checked_function.to_parsed_function()
        scope_id = self.create_scope(parent_scope_id, parsed_function.can_throw,
                                     f'function-specialization({parsed_function.name})')

        if len(parsed_function.generic_parameters) != len(generic_arguments):
            self.error(f'Generic function `{parsed_function.name}` expects '
                       f'{len(parsed_function.generic_parameters)} generic arguments, '
                       f'but {len(generic_arguments)} were given', parsed_function.name_span)

        span = parsed_function.name_span
        for key, value in generic_substitutions.items():
            key_name = self.get_type(TypeId.from_string(key))
            if key_name.variant == 'TypeVariable':
                self.add_type_to_scope(scope_id, key_name.type_name, TypeId.from_string(value), span)

        parsed_function.must_instantiate = True

        self.current_function_id = function_id
        self.typecheck_function_predecl(parsed_function, scope_id, this_type_id)
        self.typecheck_function(parsed_function, scope_id)
        self.current_function_id = None

        checked_function.is_instantiated = True
        checked_function.function_scope_id = scope_id

    # TODO: rename this when a name for this new language has been decided, assuming I decide this is not just a fork
    #       of Jakt.
    def typecheck_jakt_main(self, parsed_function: ParsedFunction):
        param_type_error = 'Main function must take a single array of strings as it\'s parameter'
        if len(parsed_function.params) > 1:
            self.error(param_type_error, parsed_function.name_span)

        if len(parsed_function.params) != 0:
            if parsed_function.params[0].variable.parsed_type.variant == 'Array':
                inner = parsed_function.params[0].variable.parsed_type.inner
                span = parsed_function.params[0].variable.parsed_type.span
                if inner.variant == 'Name':
                    if inner.name != 'String':
                        self.error(param_type_error, span)
                else:
                    self.error(param_type_error, span)
            else:
                self.error(param_type_error, parsed_function.name_span)

        return_type_error = 'Main function must return c_int'
        if parsed_function.return_type.variant not in ['Empty', 'Name']:
            self.error(return_type_error, parsed_function.return_type_span)
        elif parsed_function.return_type.variant == 'Name' and parsed_function.return_type.name != 'c_int':
            self.error(return_type_error, parsed_function.return_type.span)

    def infer_function_return_type(self, block: CheckedBlock) -> TypeId:
        if not block.statements:
            return void_type_id()
        last_statement = block.statements[-1]
        if last_statement.variant == 'Return' and last_statement.val:
            return last_statement.val.type()
        return void_type_id()

    def typecheck_function(self, parsed_function: ParsedFunction, parent_scope_id: ScopeId):
        if parsed_function.generic_parameters and not parsed_function.must_instantiate:
            return

        function_id = self.find_function_in_scope(parent_scope_id, parsed_function.name)
        if not function_id:
            self.compiler.panic('Internal error: missing previously defined function')
        if parsed_function.name == 'main':
            self.typecheck_jakt_main(parsed_function)

        checked_function = self.get_function(function_id)
        function_scope_id = checked_function.function_scope_id
        function_linkage = checked_function.linkage

        param_vars: List[CheckedVariable] = []
        module = self.current_module()
        for param in checked_function.params:
            variable = param.variable
            param_vars.append(variable)
            var_id = module.add_variable(variable)
            self.add_var_to_scope(function_scope_id, variable.name, var_id, variable.definition_span)

        # resolve concrete types
        function_return_type_id = self.typecheck_typename(parsed_function.return_type, function_scope_id, None)
        checked_function.return_type_id = function_return_type_id

        if function_return_type_id == never_type_id():
            # Allow noreturn functions to call throwing functions, they'll just be forced to crash
            scope = self.get_scope(function_scope_id)
            scope.can_throw = True

        # Todo: typecheck function block - Check if this has changed, or if this is an outdated todo.
        block = self.typecheck_block(parsed_function.block, function_scope_id, SafetyMode.Safe())

        # Typecheck return type
        function_return_type_id = self.typecheck_typename(parsed_function.return_type, function_scope_id, None)

        # Infer return type if necessary
        # If the return type is unknown, and the function starts with a return satement,
        # we infer the return type from it's expression.
        UNKNOWN_TYPE_ID = unknown_type_id()
        VOID_TYPE_ID = void_type_id()
        return_type_id = VOID_TYPE_ID
        if function_return_type_id == UNKNOWN_TYPE_ID:
            return_type_id = self.infer_function_return_type(block)
        else:
            return_type_id = self.resolve_type_var(function_return_type_id, function_scope_id)

        external_linkage = function_linkage.variant == 'External'

        if not external_linkage and not return_type_id == VOID_TYPE_ID\
                and not block.control_flow.always_transfers_control():
            if return_type_id == never_type_id() and not block.control_flow.never_returns():
                self.error('Control reaches end of never-returning function', parsed_function.name_span)
            elif not block.control_flow.never_returns():
                self.error('Control reaches end of non-void function', parsed_function.name_span)

        checked_function.block = block
        checked_function.return_type_id = return_type_id

    def statement_control_flow(self, statement: CheckedStatement) -> BlockControlFlow:
        variant = statement.variant
        if variant == 'Return':
            return BlockControlFlow.AlwaysReturns()
        elif variant == 'Throw':
            return BlockControlFlow.AlwaysReturns()
        elif variant == 'Break':
            return BlockControlFlow.AlwaysTransfersControl(might_break=True)
        elif variant == 'Continue':
            return BlockControlFlow.AlwaysTransfersControl(might_break=False)
        elif variant == 'Yield':
            return statement.expr.control_flow().updated(BlockControlFlow.AlwaysTransfersControl(might_break=False))
        elif variant == 'If':
            condition = statement.condition
            then_block = statement.then_block
            else_statement = statement.else_statement
            if condition.variant == 'Boolean':
                if condition.val:
                    return then_block.control_flow
                else:
                    if else_statement:
                        return self.statement_control_flow(else_statement)
                    else:
                        return BlockControlFlow.MayReturn()
            else:
                if then_block.control_flow.variant == 'NeverReturns':
                    return self.maybe_statement_control_flow(else_statement, then_block.control_flow)
                elif then_block.control_flow.variant == 'AlwaysReturns':
                    intermediate = self.maybe_statement_control_flow(else_statement, then_block.control_flow)
                    if intermediate.variant in ['NeverReturns', 'AlwaysReturns']:
                        return BlockControlFlow.AlwaysReturns()
                    elif intermediate.variant == 'MayReturn':
                        return BlockControlFlow.MayReturn()
                    elif intermediate.variant == 'AlwaysTransfersControl':
                        return BlockControlFlow.AlwaysTransfersControl(intermediate.might_break)
                    elif intermediate.variant == 'PartialNeverReturns':
                        return BlockControlFlow.PartialNeverReturns(intermediate.might_break)
                    elif intermediate.variant == 'PartialAlwaysReturns':
                        return BlockControlFlow.PartialAlwaysReturns(intermediate.might_break)
                    elif intermediate.variant == 'PartialAlwaysTransfersControl':
                        return BlockControlFlow.PartialAlwaysTransfersControl(intermediate.might_break)
                elif then_block.control_flow.variant == 'MayReturn':
                    return BlockControlFlow.MayReturn()
                elif then_block.control_flow.variant in ['PartialNeverReturns', 'PartialAlwaysReturns',
                                                         'PartialAlwaysTransfersControl', 'AlwaysTransfersControl']:
                    return self.maybe_statement_control_flow(else_statement, then_block.control_flow)
        elif variant == 'Block':
            return statement.block.control_flow
        elif variant == 'While':
            block_flow = statement.block.control_flow.variant
            if block_flow == 'AlwaysTransfersControl':
                return BlockControlFlow.MayReturn()
            elif block_flow == 'NeverReturns':
                return BlockControlFlow.NeverReturns()
            elif block_flow == 'AlwaysReturns':
                return BlockControlFlow.AlwaysReturns()
            else:
                return BlockControlFlow.MayReturn()
        elif variant == 'Loop':
            block_flow = statement.block.control_flow.variant
            if block_flow == 'AlwaysTransfersControl':
                return BlockControlFlow.AlwaysTransfersControl(statement.block.control_flow.might_break)
            elif block_flow == 'NeverReturns':
                return BlockControlFlow.NeverReturns()
            elif block_flow == 'AlwaysReturns':
                return BlockControlFlow.AlwaysReturns()
            elif block_flow == 'MayReturn':
                return BlockControlFlow.MayReturn()
            else:
                if statement.block.control_flow.may_break():
                    return BlockControlFlow.MayReturn()
                # Loop will always continue, so upgrade partial results to full ones
                elif block_flow == 'PartialAlwaysReturns':
                    return BlockControlFlow.AlwaysReturns()
                elif block_flow == 'PartialNeverReturns':
                    return BlockControlFlow.NeverReturns()
                elif block_flow == 'PartialAlwaysTransfersControl':
                    return BlockControlFlow.AlwaysTransfersControl(statement.block.control_flow.might_break)
                else:
                    return BlockControlFlow.MayReturn()  # unreachable
        elif variant == 'Expression':
            return statement.expr.control_flow()
        else:
            return BlockControlFlow.MayReturn()

    def maybe_statement_control_flow(self, statement: CheckedStatement | None,
                                     other_branch: BlockControlFlow) -> BlockControlFlow:
        if statement:
            return self.statement_control_flow(statement)
        else:
            return other_branch.partial()

    # FIXME: Use [TypeId: TypeId] without TypeId.to_string/from_string workaround
    def check_types_for_compat(self, lhs_type_id: TypeId, rhs_type_id: TypeId,
                               generic_inferences: Dict[str, str], span: TextSpan) -> bool:
        lhs_type = self.get_type(lhs_type_id)

        lhs_type_id_string = lhs_type_id.to_string()
        rhs_type_id_string = rhs_type_id.to_string()

        optional_struct_id = self.find_struct_in_prelude('Optional')
        weakptr_struct_id = self.find_struct_in_prelude('WeakPtr')
        array_struct_id = self.find_struct_in_prelude('Array')

        if lhs_type_id == unknown_type_id() or rhs_type_id == unknown_type_id():
            return True

        variant = lhs_type.variant
        if variant == 'TypeVariable':
            # If the call expects a generic type variable, let's see if we've already seen it
            seen_type_id_string = generic_inferences.get(lhs_type_id_string, None)
            if seen_type_id_string:
                # We've seen this type variable assigned something before
                # We should error if it's incompatible.
                if seen_type_id_string != rhs_type_id_string:
                    self.error(f'Type mismatch: expected `{self.type_name(TypeId.from_string(seen_type_id_string))}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', span)
                    return False
            else:
                generic_inferences[lhs_type_id_string] = rhs_type_id_string
        elif variant == 'GenericEnumInstance':
            lhs_enum_id = lhs_type.id_
            lhs_enum = self.get_enum(lhs_enum_id)
            lhs_args = lhs_type.args

            rhs_type_ = self.get_type(rhs_type_id)
            if rhs_type_.variant == 'GenericEnumInstance':
                rhs_enum_id = rhs_type_.id_
                rhs_args = rhs_type_.args
                if lhs_enum_id == rhs_enum_id:
                    if len(lhs_args) != len(rhs_args):
                        self.error(f'mismatched number of generic parameters for {lhs_enum.name}', span)
                        return False

                    for idx in range(len(lhs_args)):
                        if not self.check_types_for_compat(lhs_args[idx], rhs_args[idx], generic_inferences, span):
                            return False
            else:
                if rhs_type_id != lhs_type_id:
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', span)
                    return False
        elif variant == 'GenericInstance':
            lhs_struct_id = lhs_type.id_
            lhs_args = lhs_type.args

            # If lhs is T? or weak T? and rhs is T, skip type compat check
            if lhs_struct_id == optional_struct_id or lhs_struct_id == weakptr_struct_id:
                if len(lhs_args) > 0:
                    if lhs_args[0] == rhs_type_id:
                        return True

            rhs_type = self.get_type(rhs_type_id)
            if rhs_type.variant == 'GenericInstance':
                rhs_struct_id = rhs_type.id_
                if lhs_struct_id == rhs_struct_id:
                    rhs_args = rhs_type.args
                    lhs_struct = self.get_struct(lhs_struct_id)
                    if len(lhs_args) != len(rhs_args):
                        self.error(f'mismatched number of generic parameters for {lhs_struct.name}', span)
                        return False
                    for idx in range(len(rhs_args)):
                        if not self.check_types_for_compat(lhs_args[idx], rhs_args[idx], generic_inferences, span):
                            return False
                elif lhs_struct_id == array_struct_id:
                    array_value_type_id = rhs_type.args[0]
                    if array_value_type_id == unknown_type_id():
                        return True
                else:
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got {self.type_name(rhs_type_id)}', span)
                    return False
            else:
                if rhs_type_id != lhs_type_id:
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', span)
        elif variant == 'Enum':
            enum_id = lhs_type.enum_id
            if lhs_type_id == rhs_type_id:
                return True
            rhs_type = self.get_type(rhs_type_id)
            if rhs_type.variant == 'GenericEnumInstance':
                id_ = rhs_type.id_
                args = rhs_type.args
                if enum_id == id_:
                    lhs_enum = self.get_enum(enum_id)
                    if len(args) != len(lhs_enum.generic_parameters):
                        self.error(f'mismatched number of generic parameters for `{lhs_enum.name}`', span)
                        return False
                    for idx in range(len(args)):
                        if not self.check_types_for_compat(lhs_enum.generic_parameters[idx], args[idx],
                                                           generic_inferences, span):
                            return False
            elif rhs_type.variant == 'TypeVariable':
                seen_type_id_string = generic_inferences.get(rhs_type_id_string, None)
                if seen_type_id_string:
                    if seen_type_id_string != lhs_type_id_string:
                        self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                                   f'but got `{self.type_name(TypeId.from_string(seen_type_id_string))}`', span)
                        return False
                    else:
                        generic_inferences[lhs_type_id_string] = rhs_type_id_string
            else:
                if rhs_type_id != lhs_type_id:
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', span)
                    return False
        elif variant == 'Struct':
            lhs_struct_id = lhs_type.id_
            if lhs_type_id == rhs_type_id:
                return True
            rhs_type = self.get_type(rhs_type_id)
            if rhs_type.variant == 'GenericInstance':
                id_ = rhs_type.id_
                args = rhs_type.args
                if lhs_struct_id != id_:
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', span)
                    return False
                lhs_struct = self.get_struct(lhs_struct_id)
                if len(args) != len(lhs_struct.generic_parameters):
                    self.error(f'mismatched number of generic parameters for {lhs_struct.name}', span)
                    return False
                for idx in range(len(args)):
                    if not self.check_types_for_compat(lhs_struct.generic_parameters[idx], args[idx],
                                                       generic_inferences, span):
                        return False
            elif rhs_type.variant == 'TypeVariable':
                # if the call expects a generic type variable, lets see if we've already seen it
                seen_type_id_string = generic_inferences.get(rhs_type_id_string, None)
                if seen_type_id_string:
                    # we've seen this type variable assigned something before
                    # we should error if it's incompatible
                    if seen_type_id_string != lhs_type_id_string:
                        self.error(f'Type mismatch: '
                                   f'expected `{self.type_name(TypeId.from_string(seen_type_id_string))}`, '
                                   f'but got `{self.type_name(rhs_type_id)}`', span)
                        return False
                else:
                    generic_inferences[lhs_type_id_string] = rhs_type_id_string
            else:
                if rhs_type_id != lhs_type_id:
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', span)
                    return False
        elif variant == 'RawPtr':
            lhs_rawptr_type_id = lhs_type.id_
            if lhs_rawptr_type_id == rhs_type_id:
                return True

            rhs_type = self.get_type(rhs_type_id)
            if rhs_type.variant == 'RawPtr':
                if not self.check_types_for_compat(lhs_rawptr_type_id, rhs_type.id_, generic_inferences, span):
                    return False
            else:
                if rhs_type_id != lhs_type_id:
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', span)
                    return False
        else:
            if rhs_type_id != lhs_type_id:
                self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                           f'but got `{self.type_name(rhs_type_id)}`', span)
                return False
        return True

    # FIXME: use [TypeId: TypeId] without TypeId.to_string()/from_string() workaround
    def substitute_typevars_in_type(self, type_id: TypeId, generic_inferences: Dict[str, str]) -> TypeId:
        result = self.substitute_typevars_in_type_helper(type_id, generic_inferences)

        while True:
            fixed_point = self.substitute_typevars_in_type_helper(type_id, generic_inferences)

            if fixed_point == result:
                break
            else:
                result = fixed_point
        return result

    def substitute_type_vars_in_type_helper(self, type_id: TypeId, generic_inferences: Dict[str, str]) -> TypeId:
        type_ = self.get_type(type_id)

        if type_.variant == 'TypeVariable':
            replacement_type_id_string = generic_inferences.get(type_id.to_string(), None)
            if replacement_type_id_string:
                return TypeId.from_string(replacement_type_id_string)
        elif type_.variant == 'GenericInstance':
            id_ = type_.id_
            args = type_.args
            new_args: List[TypeId] = []
            for arg in args:
                new_args.append(self.substitute_typevars_in_type(arg, generic_inferences))
            return self.find_or_add_type_id(Type.GenericInstance(id_, new_args))
        elif type_.variant == 'GenericEnumInstance':
            id_ = type_.id_
            args = type_.args
            new_args: List[TypeId] = []
            for arg in args:
                new_args.append(self.substitute_typevars_in_type(arg, generic_inferences))
            return self.find_or_add_type_id(Type.GenericEnumInstance(id_, new_args))
        elif type_.variant == 'Struct':
            struct_id = type_.id_
            struct_ = self.get_struct(struct_id)
            if len(struct_.generic_parameters) > 0:
                new_args: List[TypeId] = []
                for arg in struct_.generic_parameters:
                    new_args.append(self.substitute_typevars_in_type(arg, generic_inferences))
                return self.find_or_add_type_id(Type.GenericInstance(struct_id, new_args))
        elif type_.variant == 'Enum':
            enum_id = type_.id_
            enum_ = self.get_enum(enum_id)
            if len(enum_.generic_parameters) > 0:
                new_args: List[TypeId] = []
                for arg in enum_.generic_parameters:
                    new_args.append(self.substitute_typevars_in_type(arg, generic_inferences))
                return self.find_or_add_type_id(Type.GenericEnumInstance(enum_id, new_args))
        elif type_.variant == 'RawPtr':
            rawptr_type_id = type_.id_
            rawptr_type = Type.RawPtr(self.substitute_typevars_in_type(rawptr_type_id, generic_inferences))
            return self.find_or_add_type_id(rawptr_type)
        elif type_.variant == 'Reference':
            ref_type_id = type_.id_
            ref_type = Type.Reference(self.substitute_typevars_in_type(ref_type_id, generic_inferences))
            return self.find_or_add_type_id(ref_type)
        elif type_.variant == 'MutableReference':
            ref_type_id = type_.id_
            ref_type = Type.MutableReference(self.substitute_typevars_in_type(ref_type_id, generic_inferences))
            return self.find_or_add_type_id(ref_type)
        else:
            return type_id
        return type_id

    def typecheck_block(self, parsed_block: ParsedBlock, parent_scope_id: ScopeId,
                        safety_mode: SafetyMode) -> CheckedBlock:
        parent_throws = self.get_scope(parent_scope_id).can_throw
        block_scope_id = self.create_scope(parent_scope_id, parent_throws, 'block')
        checked_block = CheckedBlock([], block_scope_id, BlockControlFlow.MayReturn(), TypeId.none())
        generic_inferences: Dict[str, str] = {}
        for parsed_statement in parsed_block.stmts:
            if checked_block.control_flow.never_returns():
                self.error('Unreachable code', parsed_statement.span)

            checked_statement = self.typecheck_statement(parsed_statement, block_scope_id, safety_mode)
            checked_block.control_flow = checked_block.control_flow.updated(
                    self.statement_control_flow(checked_statement))
            yield_span: TextSpan | None = None
            if parsed_statement.variant == 'Yield':
                yield_span = parsed_statement.expr.span
            checked_yield_expression: CheckedExpression | None = None
            if checked_statement.variant == 'Yield':
                checked_yield_expression = checked_statement.expr
            if yield_span and checked_yield_expression:
                type_var_type_id = checked_yield_expression.type()
                type_ = self.resolve_type_var(type_var_type_id, block_scope_id)
                if checked_block.yielded_type:
                    # TODO: check types for compat - isn't this already done here?
                    self.check_types_for_compat(checked_block.yielded_type, type_, generic_inferences, yield_span)
                else:
                    checked_block.yielded_type = type_

            checked_block.statements.append(checked_statement)

        if checked_block.yielded_type:
            checked_block.yielded_type = self.substitute_typevars_in_type(checked_block.yielded_type, generic_inferences)
        return checked_block

    def typecheck_typename(self, parsed_type: ParsedType, scope_id: ScopeId, name: str | None) -> TypeId:
        print(f'typecheck_typename: parsed_type: {parsed_type}, name: {name}')
        if parsed_type.variant == 'Reference':
            inner_type_id = self.typecheck_typename(parsed_type.inner, scope_id, name)
            type_id = self.find_or_add_type_id(Type.Reference(inner_type_id))
            return type_id
        elif parsed_type.variant == 'MutableReference':
            inner_type_id = self.typecheck_typename(parsed_type.inner, scope_id, name)
            type_id = self.find_or_add_type_id(Type.MutableReference(inner_type_id))
            return type_id
        elif parsed_type.variant == 'NamespacedName':
            current_namespace_scope_id = scope_id
            for ns in parsed_type.namespaces:
                result, imported = self.find_namespace_in_scope(current_namespace_scope_id, ns)
                if result:
                    current_namespace_scope_id = result
                else:
                    self.error(f'Unknown namespace: `{ns}`', parsed_type.span)
                    return unknown_type_id()

            generic_args: List[TypeId] = []

            for param in parsed_type.params:
                checked_arg = self.typecheck_typename(param, scope_id, name)
                generic_args.append(checked_arg)

            if len(generic_args) == 0:
                synthetic_typename = ParsedType.Name(name, parsed_type.span)
                return self.typecheck_typename(synthetic_typename, current_namespace_scope_id, name)
            else:
                return self.typecheck_generic_resolved_type(name, generic_args, current_namespace_scope_id, parsed_type.span)
        elif parsed_type.variant == 'Name':
            print(f'typecheck_typename: looking in {scope_id} for type name `{parsed_type.name}` with name: {name}')
            result = self.find_type_scope(scope_id, name)
            print(f'typecheck_typename: result: {result}')
            if result:
                maybe_type, maybe_scope = result
                print(f'maybe_type: {maybe_type}, maybe_scope: {maybe_scope}')
                if maybe_scope != self.prelude_scope_id():
                        print(f'found {maybe_type} in prelude scope')
                        return maybe_type
                else:
                    self.compiler.panic(f'typecheck_typename: maybe_scope is not the prelude scope, expected ScopeId(id_=0), got `{scope_id}`')

            print(f'matching {parsed_type.name} against builtins')
            match parsed_type.name:
                case 'i8':
                    return builtin(BuiltinType.I8())
                case 'i16':
                    return builtin(BuiltinType.I16())
                case 'i32':
                    return builtin(BuiltinType.I32())
                case 'i64':
                    return builtin(BuiltinType.I64())
                case 'u8':
                    return builtin(BuiltinType.U8())
                case 'u16':
                    return builtin(BuiltinType.U16())
                case 'u32':
                    return builtin(BuiltinType.U32())
                case 'u64':
                    return builtin(BuiltinType.U64())
                case 'f32':
                    return builtin(BuiltinType.F32())
                case 'f64':
                    return builtin(BuiltinType.F64())
                case 'c_char':
                    return builtin(BuiltinType.CChar())
                case 'c_int':
                    return builtin(BuiltinType.CInt())
                case 'usize':
                    return builtin(BuiltinType.Usize())
                case 'String':
                    return builtin(BuiltinType.String())
                case 'bool':
                    return builtin(BuiltinType.Bool())
                case 'void':
                    return builtin(BuiltinType.Void())
                case 'never':
                    return builtin(BuiltinType.Never())
                case _:

                    if result and result[1]:
                        print(f'typecheck_typename: didn\'t match against known builtin, returning {result[1]}')
                        return result[1]
                    else:
                        self.error(f'Unknown type `{name}`', parsed_type.span)
                        print(f'typecheck_typename: Unknown type `{name}: {parsed_type.name}`, returning unknown_type_id()')
                        return unknown_type_id()
        elif parsed_type.variant == 'Empty':
            print('typecheck_typename: converting Empty type to unknown type')
            return unknown_type_id()
        elif parsed_type.variant == 'Tuple':
            print('typecheck_typename: Checking Tuple types')
            print(f'typecheck_typename: parsed_type: {parsed_type}')
            checked_types: List[TypeId] = []

            print('typecheck_typename: before loop')
            for parsed_type_ in parsed_type.types:
                print(f'typecheck_typename: typechecking typename `{parsed_type_}`')
                result = self.typecheck_typename(parsed_type_, scope_id, name)
                print(f'typecheck_typename: appending `{result}` to checked_types')
                checked_types.append(result)
            print('typecheck_typename: after loop')
            tuple_struct_id = self.find_struct_in_prelude("Tuple")
            print(f'typecheck_typename: tuple_struct_id: {tuple_struct_id}')
            print(f'typecheck_typename: checked_types length: {len(checked_types)}')
            print(f'typecheck_typename: checked_types: {checked_types}')
            type_id = self.find_or_add_type_id(Type.GenericInstance(tuple_struct_id, checked_types))
            print(f'typecheck_typename: type_id: {type_id}')
            return type_id
        elif parsed_type.variant == 'Array':
            print('typecheck_typename: looking up Array type...')
            inner_type_id = self.typecheck_typename(parsed_type.inner, scope_id, name)
            print(f'inner_type_id: {inner_type_id}')
            array_struct_id = self.find_struct_in_prelude('Array')
            print(f'array_struct_id: {array_struct_id}')
            type_id = self.find_or_add_type_id(Type.GenericInstance(array_struct_id, [inner_type_id]))
            print(f'type_id: {type_id}')
            return type_id
        elif parsed_type.variant == 'Dictionary':
            print('typecheck_typename: Checking Dictionary types')
            key_type_id = self.typecheck_typename(parsed_type.key, scope_id, name)
            value_type_id = self.typecheck_typename(parsed_type.value, scope_id, name)
            dict_struct_id = self.find_struct_in_prelude("Dictionary")
            type_id = self.find_or_add_type_id(Type.GenericInstance(dict_struct_id, [key_type_id, value_type_id]))
            return type_id
        elif parsed_type.variant == 'Set':
            print('typecheck_typename: Checking Set types')
            inner_type_id = self.typecheck_typename(parsed_type.inner, scope_id, name)
            set_struct_id = self.find_struct_in_prelude('Set')
            type_id = self.find_or_add_type_id(Type.GenericInstance(set_struct_id, [inner_type_id]))
            return type_id
        elif parsed_type.variant == 'Optional':
            print('typecheck_typename: Checking Optional types')
            inner_type_id = self.typecheck_typename(parsed_type.inner, scope_id, name)
            optional_struct_id = self.find_struct_in_prelude('Optional')
            type_id = self.find_or_add_type_id(Type.GenericInstance(optional_struct_id, [inner_type_id]))
            return type_id
        elif parsed_type.variant == 'WeakPtr':
            print('typecheck_typename: Checking WeakPtr types')
            inner_type_id = self.typecheck_typename(parsed_type.inner, scope_id, name)
            weakptr_struct_id = self.find_struct_in_prelude('WeakPtr')
            type_id = self.find_or_add_type_id(Type.GenericInstance(weakptr_struct_id, [inner_type_id]))
            return type_id
        elif parsed_type.variant == 'RawPointer':
            print('typecheck_typename: Checking RawPointer types')
            print(f'typecheck_typename: name: {name}, RawPointer, parsed_type:')
            pprint(parsed_type)
            inner_type_id = self.typecheck_typename(parsed_type.inner, scope_id, name)
            print(f'inner_type_id = {inner_type_id}')
            type_id = self.find_or_add_type_id(Type.RawPtr(inner_type_id))
            return type_id
        elif parsed_type.variant == 'GenericType':
            print('typecheck_typename: Checking Generic parameters')
            checked_inner_types: List[TypeId] = []
            for inner_type in parsed_type.generic_parameters:
                inner_type_id = self.typecheck_typename(inner_type, scope_id, name)
                checked_inner_types.append(inner_type_id)
            return self.typecheck_generic_resolved_type(name, checked_inner_types, scope_id, parsed_type.span)
        elif parsed_type.variant == 'Function':
            print('typecheck_typename: Checking Function type')
            if name:
                function_name = name
            else:
                function_name = f'lambda{self.lambda_count}'
                self.lambda_count += 1
            checked_params: List[CheckedParameter] = []
            first = True
            for param in parsed_type.params:
                checked_params.append(self.typecheck_parameter(param, scope_id, first, None, None))
                first = False

            checked_function = CheckedFunction(function_name, parsed_type.span, Visibility.Public(),
                                               self.typecheck_typename(parsed_type.return_type, scope_id, None),
                                               parsed_type.return_type.span, checked_params, [],
                                               CheckedBlock([], scope_id, BlockControlFlow.MayReturn(), None),
                                               parsed_type.can_throw, FunctionType.Normal(), FunctionLinkage.Internal(),
                                               scope_id, False, None, False)
            module = self.current_module()
            function_id = module.add_function(checked_function)
            self.add_function_to_scope(scope_id, checked_function.name, function_id, parsed_type.span)
            param_type_ids: List[TypeId] = []
            for param in parsed_type.params:
                param_type_ids.append(self.typecheck_typename(param.variable.parsed_type, scope_id, name))
            return_type_id = self.typecheck_typename(parsed_type.return_type, scope_id, name)
            return self.find_or_add_type_id(
                    Type.Function(param_type_ids, parsed_type.can_throw, parsed_type.return_type_id))
        else:
            self.compiler.panic(f'typecheck_typename: Unreachable while checking parsed_type.variant. parsed_type.variant = {parsed_type.variant}')

    def typecheck_generic_resolved_type(self, name: str, checked_inner_types: List[TypeId], scope_id: ScopeId,
                                        span: TextSpan) -> TypeId:
        struct_id = self.find_struct_in_scope(scope_id, name)
        if struct_id:
            return self.find_or_add_type_id(Type.GenericInstance(struct_id, checked_inner_types))

        enum_id = self.program.find_enum_in_scope(scope_id, name)
        if enum_id:
            return self.find_or_add_type_id(Type.GenericInstance(enum_id, checked_inner_types))

        self.error(f'could not find {name}', span)
        return unknown_type_id()

    def typecheck_unary_operation(self, checked_expr: CheckedExpression, checked_op: CheckedUnaryOperator,
                                  span: TextSpan, scope_id: ScopeId, safety_mode: SafetyMode):
        expr_type_id = checked_expr.type()
        expr_type = self.get_type(expr_type_id)

        match checked_op.variant:
            case 'PreIncrement', 'PostIncrement', 'PreDecrement', 'PostDecrement':
                if self.is_integer(expr_type_id):
                    if not checked_expr.is_mutable(self.program):
                        self.error("Increment/decrement of immutable variable", span)
                    else:
                        self.error("Increment/decrement of non-numeric value", span)
            case 'LogicalNot', 'BitwiseNot':
                return CheckedExpression.UnaryOp(checked_expr, checked_op, span, expr_type_id)
            case 'TypeCast':
                return CheckedExpression.UnaryOp(checked_expr, checked_op, span, checked_op.cast.type_id())
            case 'Negate':
                return self.typecheck_unary_negate(checked_expr, span, expr_type_id)
            case 'Is', 'IsEnumVariant':
                return CheckedExpression.UnaryOp(checked_expr, checked_op, span, builtin(BuiltinType.Bool()))
            case 'RawAddress':
                return CheckedExpression.UnaryOp(checked_expr, checked_op, span, self.find_or_add_type_id(Type.RawPtr(expr_type_id)))
            case 'Reference':
                return CheckedExpression.UnaryOp(checked_expr, checked_op, span, self.find_or_add_type_id(Type.Reference(expr_type_id)))
            case 'MutableReference':
                if not checked_expr.is_mutable(self.program):
                    self.error("Cannot make mutable reference to immutable value", span)
                return CheckedExpression.UnaryOp(checked_expr, checked_op, span, self.find_or_add_type_id(Type.MutableReference(expr_type_id)))
            case 'Dereference':
                match expr_type:
                    case 'RawPtr':
                        if safety_mode.variant == 'Safe':
                            self.error("Dereference of raw pointer outside of unsafe block", span)
                        return CheckedExpression.UnaryOp(checked_expr, checked_op, span, expr_type.type_id)
                    case 'Reference', 'MutableReference':
                        return CheckedExpression.UnaryOp(checked_expr, checked_op, span, expr_type.type_id)
                    case _:
                        self.error("Dereference of a non-pointer value", span)

        return CheckedExpression.UnaryOp(checked_expr, checked_op, span, expr_type_id)

    def typecheck_unary_negate(self, expr: CheckedExpression, span: TextSpan, type_id: TypeId) -> CheckedExpression:
        if not self.program.is_integer(type_id) or self.program.is_signed(type_id):
            return CheckedExpression.UnaryOp(expr, CheckedUnaryOperator.Negate(), span, type_id)

        # Flipping the sign on a small enough unsigned constant is fine. We'll change the type to the signed variant.
        flipped_sign_type = flip_signedness(self.get_type(type_id))

        constant: CheckedNumericConstant | None
        if expr.variant == 'NumericConstant':
            constant = expr.val
        else:
            return CheckedExpression.UnaryOp(expr, CheckedUnaryOperator.Negate(), span, type_id)

        if constant:
            number = constant.number_constant()
            raw_number = number.to_usize()
            max_signed = Type.I64().max()
            negated_number: int = 0
            if raw_number == max_signed + 1:
                negated_number = Type.I64().min()
            if raw_number <= max_signed:
                negated_number = 0 - raw_number
            negated_number_constant = NumberConstant.Signed(c_long(negated_number))
            if raw_number > (max_signed + 1) or \
                    not negated_number_constant.can_fit_number(flipped_sign_type, self.program):
                self.error(f'Negative literal -{raw_number} too small for type `{self.type_name(flipped_sign_type)}`',
                           span)
                return CheckedExpression.UnaryOp(expr, CheckedUnaryOperator.Negate(), span, type_id)

            new_constant: CheckedNumericConstant | None = None
            match self.get_type(flipped_sign_type).variant:
                case 'I8': new_constant = CheckedNumericConstant.I8(c_byte(negated_number))
                case 'I16': new_constant = CheckedNumericConstant.I16(c_short(negated_number))
                case 'I32': new_constant = CheckedNumericConstant.I32(c_int(negated_number))
                case 'I64': new_constant = CheckedNumericConstant.I64(c_long(negated_number))
                case _:
                    self.compiler.panic('typecheck_unary_negate: Unreachable')
            if not new_constant:
                self.compiler.panic('typecheck_unary_negate: new_constant is still None, this should be impossible')

            return CheckedExpression.UnaryOp(CheckedExpression.NumericConstant(new_constant, span, type_id),
                                             CheckedUnaryOperator.Negate(), span, flipped_sign_type)

    def typecheck_binary_operation(self, checked_lhs: CheckedExpression, op: BinaryOperator,
                                   checked_rhs: CheckedExpression, scope_id: ScopeId, span: TextSpan) -> TypeId:
        lhs_type_id = checked_lhs.type()
        rhs_type_id = checked_rhs.type()

        lhs_span = checked_lhs.span
        rhs_span = checked_rhs.span

        type_id = checked_lhs.type()

        match op.variant:
            case 'NoneCoalescing', 'NoneCoalescingAssign':
                # 1. LHS must be Optional<T>.
                # 2. RHS must be Optional<T> or T.
                # 3. Resulting type is Optional<T> or T, respectively.

                # if an assignment, the LHS must be a mutable variable.
                if op.variant == 'NoneCoalescingAssign':
                    if checked_lhs.variant == 'Var':
                        if not checked_lhs.var.is_mutable:
                            self.error_with_hint('left-hand side of ??= must be a mutable variable', checked_lhs.span,
                                                 'This variable isn\'t marked as mutable',
                                                 checked_lhs.var.definition_span)
                            return unknown_type_id()
                    else:
                        self.error('left-hand side of ??= must be a mutable variable', checked_lhs.span)
                        return unknown_type_id()
                lhs_type = self.get_type(lhs_type_id)
                if lhs_type.variant == 'GenericInstance' and lhs_type.id_ == self.find_struct_in_prelude('Optional'):
                    # Success: LHS is T? and RHS is T?.
                    if lhs_type_id == rhs_type_id:
                        return lhs_type_id

                    # Extract T from Optional<T>.
                    inner_type_id = lhs_type.args[0]

                    if inner_type_id == rhs_type_id:
                        # Success: LHS is T? and RHS is T.
                        return inner_type_id
                else:
                   self.error_with_hint(f'None coalescing (??) with incompatible types '
                                        f'(`{self.type_name(lhs_type_id)}` and `{self.type_name(rhs_type_id)}`)',
                                        span, 'Left side of ?? must be an Optional but isn\'t', lhs_span)

                self.error(f'None coalescing (??) with incompatible types (`{self.type_name(lhs_type_id)}` and '
                           f'`{self.type_name(rhs_type_id)}`)', span)
                return lhs_type_id
            case 'LessThan', 'LessThanOrEqual', 'GreaterThan', 'GreaterThanOrEqual', 'Equal', 'NotEqual':
                if lhs_type_id != rhs_type_id:
                    self.error(f'Binary comparison between incompatible types '
                               f'(`{self.type_name(lhs_type_id)}` and `{self.type_name(rhs_type_id)}`)', span)
                type_id = builtin(BuiltinType.Bool())
            case 'LogicalAnd', 'LogicalOr':
                if lhs_type_id != builtin(BuiltinType.Bool()):
                    self.error('left side of logical binary operation is not a boolean', lhs_span)
                if rhs_type_id != builtin(BuiltinType.Bool()):
                    self.error('right side of logical binary operation is not a boolean', rhs_span)

                type_id = builtin(BuiltinType.Bool())
            case 'Assign':
                if not checked_lhs.is_mutable(self.program):
                    self.error('Assignment to immutable variable', span)
                    return lhs_type_id
                if checked_rhs.variant == 'OptionalNone':
                    lhs_type = self.get_type(lhs_type_id)
                    if lhs_type.variant == 'GenericInstance' and lhs_type.id_ == self.find_struct_in_prelude('Optional'):
                        return lhs_type_id

                lhs_type = self.get_type(lhs_type_id)
                if lhs_type.variant == 'GerenicInstance':
                    if self.program.get_struct(lhs_type.id_).name == 'Optional' and checked_rhs.type() == lhs_type.args[0]:
                        return lhs_type_id
                    if self.program.get_struct(lhs_type.id_).name == 'WeakPtr' and checked_rhs.type() == lhs_type.args[0]:
                        return lhs_type_id

                # NOTE: Pay attention, the order is flipped here
                result = self.unify(rhs_type_id, rhs_span, lhs_type_id, lhs_span)
                if not result:
                    self.error(f'Assignment between incompatible types '
                               f'(‘{self.type_name(lhs_type_id)}’ and ‘{self.type_name(rhs_type_id)}’)', span)
                return result if result else lhs_type_id
            case 'AddAssign', 'SubtractAssign', 'MultiplyAssign', 'DivideAssign', \
                 'ModuloAssign', 'BitwiseAndAssign', 'BitwiseOrAssign', 'BitwiseXorAssign', \
                 'BitwiseLeftShiftAssign', 'BitwiseRightShiftAssign':
                weak_ptr_struct_id = self.find_struct_in_prelude("WeakPtr")
                lhs_type = self.get_type(lhs_type_id)
                lhs_arg_type = self.get_type(lhs_type.args[0])
                rhs_type = self.get_type(rhs_type_id)
                if lhs_type.variant == 'GenericInstance' and lhs_type.id_ == weak_ptr_struct_id:
                    if lhs_arg_type.variant == 'Struct':
                        lhs_struct_id = lhs_arg_type.id_
                        if rhs_type.variant == 'Struct':
                            rhs_struct_id = rhs_type.id_
                            if lhs_struct_id == rhs_struct_id:
                                return lhs_struct_id

                if lhs_type_id != rhs_type_id:
                    self.error(f'Assignment between incompatible types '
                               f'(‘{self.type_name(lhs_type_id)}’ and ‘{self.type_name(rhs_type_id)}’)', span)
                if not checked_lhs.is_mutable(self.program):
                    self.error('Assignment to immutable variable', span)
            case 'Add', 'Subtract', 'Multiply', 'Divide', 'Modulo':
                if lhs_type_id != rhs_type_id:
                    self.error(f'Binary arithmetic operation between incompatible types '
                               f'(‘{self.type_name(lhs_type_id)}’ and ‘{self.type_name(rhs_type_id)}’)', span)
                type_id = lhs_type_id
        return type_id

    def typecheck_statement(self, statement: ParsedStatement, scope_id: ScopeId,
                            safety_mode: SafetyMode) -> CheckedStatement:
        match statement.variant:
            case 'Expression':
                return CheckedStatement.Expression(self.typecheck_expression(statement.expr, scope_id, safety_mode,
                                                                             TypeId.none()), statement.statement.span)
            case 'UnsafeBlock':
                return CheckedStatement.Block(self.typecheck_block(statement.block,  scope_id,  SafetyMode.Unsafe()),
                                              statement.span)
            case 'Yield':
                return CheckedStatement.Yield(self.typecheck_expression(statement.expr, scope_id, safety_mode,
                                                                        TypeId.none()), statement.span)
            case 'Return':
                return self.typecheck_return(statement.expr, statement.span, scope_id, safety_mode)
            case 'Block':
                return self.typecheck_block_statement(statement.block, scope_id, safety_mode, statement.span)
            case 'InlineCpp':
                return self.typecheck_inline_cpp(statement.block, statement.span, safety_mode)
            case 'Defer':
                return self.typecheck_defer(statement, scope_id, safety_mode, statement.span)
            case 'Loop':
                return self.typecheck_loop(statement.block, scope_id, safety_mode, statement.span)
            case 'Throw':
                return self.typecheck_throw(statement.expr, scope_id, safety_mode, statement.span)
            case 'While':
                return self.typecheck_while(statement.condition, statement.block, scope_id, safety_mode, statement.span)
            case 'Continue':
                return CheckedStatement.Continue(statement.span)
            case 'Break':
                return CheckedStatement.Break(statement.span)
            case 'VarDecl':
                return self.typecheck_var_decl(statement.var, statement.init, scope_id, safety_mode, statement.span)
            case 'DestructuringAssignment':
                return self.typecheck_destructuring_assignment(statement.vars_, statement.var_decl, scope_id,
                                                               safety_mode, statement.span)
            case 'If':
                return self.typecheck_if(statement.condition, statement.then_block, statement.else_statement, scope_id,
                                         safety_mode, statement.span)
            case 'Invalid':
                return CheckedStatement.Invalid(statement.span)
            case 'For':
                return self.typecheck_for(statement.iterator_name, statement.name_span, statement.range_,
                                          statement.block, scope_id, safety_mode, statement.span)
            case 'Guard':
                return self.typecheck_guard(statement.expr, statement.else_block, statement.remaining_code, scope_id,
                                            safety_mode, statement.span)

    def typecheck_guard(self, expr: ParsedExpression, else_block: ParsedBlock, remaining_code: ParsedBlock,
                        scope_id: ScopeId, safety_mode: SafetyMode, span: TextSpan) -> CheckedStatement:
        seen_scope_exit: bool = False
        for statement in else_block.stmts:
            if statement.variant in ['Break', 'Continue', 'Return', 'Throw']:
                seen_scope_exit = True
                break

        # Ensure we dont use any bindings we shouldn't have access to
        checked_else_block = self.typecheck_block(else_block, scope_id, safety_mode)

        if not seen_scope_exit and checked_else_block.control_flow.may_return():
            self.error('Else block of guard must either `return`, `break`, `continue`, or `throw`', span)

        new_condition, new_then_block, new_else_statement = self.expand_context_for_bindings(
                expr, None, remaining_code, ParsedStatement.Block(else_block, span), span)
        checked_condition = self.typecheck_expression_and_dereference_if_needed(
                new_condition, scope_id, safety_mode, None, span)
        if not checked_condition.type() == builtin(BuiltinType.Bool()):
            self.error('Condition must be a boolean expression', new_condition.span)

        checked_block = self.typecheck_block(new_then_block, scope_id, safety_mode)
        if checked_block.yielded_type:
            self.error('A `guard` block is not allowed to yield values', new_then_block.find_yield_span())

        checked_else: CheckedStatement | None = None
        if new_else_statement:
            checked_else = self.typecheck_statement(new_else_statement, scope_id, safety_mode)

        return CheckedStatement.If(checked_condition, checked_block, checked_else, span)

    def typecheck_for(self, iterator_name: str, name_span: TextSpan, range_: ParsedExpression, block: ParsedBlock,
                      scope_id: ScopeId, safety_mode: SafetyMode, span: TextSpan) -> CheckedStatement:
        maybe_span = block.find_yield_span()
        if maybe_span:
            self.error('a `for` loop block is not allowed to yield values', maybe_span)

        # Translate `for x in expr { body }` to
        # block {
        #     let (mutable) _magic = expr
        #     loop {
        #         let x = _magic.next()
        #         if not x.has_value() {
        #             break
        #         }
        #         let iterator_name = x!
        #         body
        #     }
        # }
        #
        # The only restrictions placed on the iterator are such:
        #     1- Must respond to .next(); the mutability of the iterator is inferred from .next()'s signature
        #     2- The result of .next() must be an Optional.

        iterable_expr = self.typecheck_expression(range_, scope_id, safety_mode, None)
        iterable_should_be_mutable = False

        iterable_type = self.program.get_type(iterable_expr.type())

        match iterable_type.variant:
            case 'TypeVariable':
                # Since we're not sure, just make it mutable
                iterable_should_be_mutable = True
            case 'GenericInstance' | 'Struct':
                struct_ = self.get_struct(iterable_type.id_)
                next_method_function_id = self.find_function_in_scope(struct_.scope_id, 'next')
                if not next_method_function_id:
                    self.error('Iterator must have a .next() method', range_.span)
                else:
                    next_method_function = self.get_function(next_method_function_id)
                    # Check whether we need to make the iterator mutable
                    if next_method_function.is_mutating():
                        iterable_should_be_mutable = True
            case _:
                self.error('Iterator must have a .next() method', name_span)

        rewritten_statement = ParsedStatement.Block(
                block=ParsedBlock(
                        stmts=[
                                # let (mutable) _magic = expr
                                ParsedStatement.VarDecl(
                                        var=ParsedVarDecl(
                                                name='_magic',
                                                parsed_type=ParsedType.Empty(),
                                                is_mutable=iterable_should_be_mutable,
                                                inlay_span=None,
                                                span=name_span
                                                ),
                                        init=range_,
                                        span=span
                                        ),
                                ParsedStatement.Loop(
                                        block=ParsedBlock(
                                                stmts=[
                                                        # let _magic_value = _magic.next()
                                                        ParsedStatement.VarDecl(
                                                                var=ParsedVarDecl(
                                                                        name='_magic_value',
                                                                        parsed_type=ParsedType.Empty(),
                                                                        is_mutable=iterable_should_be_mutable,
                                                                        inlay_span=None,
                                                                        span=name_span
                                                                        ),
                                                                init=ParsedExpression.MethodCall(
                                                                        expr=ParsedExpression.Var(
                                                                                name='_magic',
                                                                                span=name_span
                                                                                ),
                                                                        call=ParsedCall(
                                                                                namespace=[],
                                                                                name='next',
                                                                                args=[],
                                                                                type_args=[]
                                                                                ),
                                                                        is_optional=False,
                                                                        span=name_span
                                                                        ),
                                                                span=span
                                                                ),
                                                        # if not _magic_value.has_value() {
                                                        ParsedStatement.If(
                                                                condition=ParsedExpression.UnaryOp(
                                                                        expr=ParsedExpression.MethodCall(
                                                                                expr=ParsedExpression.Var(
                                                                                        name='_magic_value',
                                                                                        span=name_span
                                                                                        ),
                                                                                call=ParsedCall(
                                                                                        namespace=[],
                                                                                        name='has_value',
                                                                                        args=[],
                                                                                        type_args=[]
                                                                                        ),
                                                                                is_optional=False,
                                                                                span=name_span
                                                                                ),
                                                                        op=UnaryOperator.LogicalNot(),
                                                                        span=name_span
                                                                        ),
                                                                then_block=ParsedBlock(
                                                                        stmts=[
                                                                                # break
                                                                                ParsedStatement.Break(span)
                                                                                ]
                                                                        ),
                                                                else_statement=None,
                                                                span=span
                                                                ),
                                                                # let iterator_name = _magic_value!
                                                                ParsedStatement.VarDecl(
                                                                        var=ParsedVarDecl(
                                                                                name=iterator_name,
                                                                                parsed_type=ParsedType.Empty(),
                                                                                # FIXME: loop variable mutability should
                                                                                #  be independent of iterable mutability
                                                                                is_mutable=iterable_should_be_mutable,
                                                                                inlay_span=name_span,
                                                                                span=name_span
                                                                                ),
                                                                                init=ParsedExpression.ForcedUnwrap(
                                                                                    expr=ParsedExpression.Var(
                                                                                            name='_magic_value',
                                                                                            span=name_span
                                                                                    ),
                                                                                span=name_span
                                                                        ),
                                                                        span=span
                                                                ),
                                                                ParsedStatement.Block(block, span)
                                                        ]
                                                ),
                                                span=span
                                        )
                                ]
                        ),
                span=span
                )
        return self.typecheck_statement(rewritten_statement, scope_id, safety_mode)

    def expand_context_for_bindings(self, condition: ParsedExpression, acc: ParsedExpression | None,
                                    then_block: ParsedBlock, else_statement: ParsedStatement | None,
                                    span: TextSpan) -> Tuple[ParsedExpression, ParsedBlock, ParsedStatement | None]:
        if condition.variant == 'BinaryOp':
            if condition.op.variant == 'LogicalAnd':
                rhs_condition, rhs_then_block, rhs_else_statement = self.expand_context_for_bindings(condition.rhs, acc, then_block, else_statement, span)
                accumulated_condition = rhs_condition
                return self.expand_context_for_bindings(condition.lhs, accumulated_condition, rhs_then_block, rhs_else_statement, span)
        elif condition.variant == 'UnaryOp':
            expr = condition.expr
            op = condition.op
            if op.variant == 'IsEnumVariant':
                inner = op.inner
                bindings = op.bindings
                unary_op_single_condition = ParsedExpression.UnaryOp(expr, UnaryOperator.Is(inner), span)
                outer_if_stmts: List[ParsedStatement] = []
                for binding in bindings:
                    var = ParsedVarDecl(binding.binding, ParsedType.Empty(), False, None, binding.span)
                    enum_variant_arg = ParsedExpression.EnumVariantArg(expr, binding, inner, span)
                    outer_if_stmts.append(ParsedStatement.VarDecl(var, enum_variant_arg, span))
                inner_condition = condition
                new_then_block = then_block
                new_else_statement = else_statement
                if acc:
                    inner_condition = acc
                    outer_if_stmts.append(ParsedStatement.If(inner_condition, then_block, else_statement, span))
                else:
                    for stmt in then_block.stmts:
                        outer_if_stmts.append(stmt)
                new_then_block = ParsedBlock(outer_if_stmts)
                return self.expand_context_for_bindings(unary_op_single_condition, None, new_then_block, new_else_statement, span)
        base_condition = condition
        if acc:
            base_condition = ParsedExpression.BinaryOp(condition, BinaryOperator.LogicalAnd(), acc, span)
        return base_condition, then_block, else_statement

    def typecheck_if(self, condition: ParsedExpression, then_block: ParsedBlock, else_statement: ParsedStatement | None,
                     scope_id: ScopeId, safety_mode: SafetyMode, span: TextSpan) -> CheckedStatement:
        new_condition, new_then_block, new_else_statement = self.expand_context_for_bindings(condition, None, then_block, else_statement, span)
        checked_condition = self.typecheck_expression_and_dereference_if_needed(new_condition, scope_id, safety_mode, None, span)
        if checked_condition.type() != builtin(BuiltinType.Bool()):
            self.error('Condition must be a boolean expression', new_condition.span)

        checked_block = self.typecheck_block(new_then_block, scope_id, safety_mode)
        if checked_block.yielded_type:
            self.error('An `if` block is not allowed to yield values', new_then_block.find_yield_span())

        checked_else: CheckedStatement | None = None
        if new_else_statement:
            checked_else = self.typecheck_statement(new_else_statement, scope_id, safety_mode)
        return CheckedStatement.If(checked_condition, checked_block, checked_else, span)

    def typecheck_destructuring_assignment(self, vars: List[ParsedVarDecl], var_decl: ParsedStatement, scope_id: ScopeId,
                                           safety_mode: SafetyMode, span: TextSpan) -> CheckedStatement:
        var_decls: List[CheckedStatement] = []
        checked_tuple_var_decl = self.typecheck_statement(var_decl, scope_id, safety_mode)
        expr_type_id = unknown_type_id()
        tuple_var_id = VarId(ModuleId(0), 0)
        if checked_tuple_var_decl.variant == 'VarDecl':
            expr_type_id = checked_tuple_var_decl.init.type()
            tuple_var_id = checked_tuple_var_decl.var_id
        else:
            self.error('Destructuring assignment should be a variable declaration', span)

        inner_types: List[TypeId] = []
        tuple_type = self.get_type(expr_type_id)
        if tuple_type.variant == 'GenericInstance':
            inner_types = tuple_type.args
        else:
            self.error('Tuple Type should be Generic Instance', span)
        tuple_variable = self.program.get_variable(tuple_var_id)
        if len(vars) == len(inner_types):
            for i in range(0, len(vars)):
                new_var = vars[i]
                new_var.parsed_type = ParsedType.Name(self.type_name(inner_types[i]), span)
                init = ParsedExpression.IndexedTuple(
                        expr=ParsedExpression.Var(tuple_variable.name, span),
                        index=i,
                        is_optional=False,
                        span=span)
                var_decls.append(self.typecheck_var_decl(vars[i], init, scope_id, safety_mode, span))
        else:
            self.error('Tuple inner types should have the same size as tuple members', span)

        return CheckedStatement.DestructuringAssignment(var_decls, checked_tuple_var_decl, span)

    def typecheck_var_decl(self, var: ParsedVarDecl, init: ParsedExpression, scope_id: ScopeId, safety_mode: SafetyMode,
                           span: TextSpan) -> CheckedStatement:
        lhs_type_id = self.typecheck_typename(var.parsed_type, scope_id, var.name)
        checked_expr = self.typecheck_expression(init, scope_id, safety_mode, lhs_type_id)
        rhs_type_id = checked_expr.type()

        if lhs_type_id == unknown_type_id() and rhs_type_id != unknown_type_id():
            lhs_type_id = rhs_type_id

        promoted_rhs = self.try_to_promote_constant_expr_to_type(lhs_type_id, checked_expr, init.span)
        if promoted_rhs:
            checked_expr = promoted_rhs

        weak_ptr_struct_id = self.find_struct_in_prelude('WeakPtr')
        optional_struct_id = self.find_struct_in_prelude('Optional')

        lhs_type = self.get_type(lhs_type_id)

        self.check_that_type_doesnt_contain_reference(lhs_type_id, span)

        if lhs_type.variant == 'GenericInstance':
            if lhs_type.id_ == weak_ptr_struct_id:
                if not var.is_mutable:
                    self.error('Weak reference must be mutable', var.span)
                if lhs_type_id != rhs_type_id and lhs_type.args[0] != rhs_type_id and rhs_type_id != unknown_type_id():
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', checked_expr.span)
            elif lhs_type.id_ == optional_struct_id:
                if lhs_type_id != rhs_type_id and lhs_type.args[0] != rhs_type_id and rhs_type_id != unknown_type_id():
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', checked_expr.span)
            else:
                if lhs_type_id != rhs_type_id and rhs_type_id != unknown_type_id():
                    self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`, '
                               f'but got `{self.type_name(rhs_type_id)}`', checked_expr.span)
        elif lhs_type.is_builtin():
            number_constant = checked_expr.to_number_constant(self.program)

            is_rhs_zero = False
            if number_constant:
                match number_constant.variant:
                    case 'Signed':
                        is_rhs_zero = number_constant.val == c_long(0)
                    case 'Unsigned':
                        is_rhs_zero = number_constant.val == c_ulong(0)
                    case 'Floating':
                        is_rhs_zero = number_constant.val == c_double(0.0)

            if not (self.is_numeric(lhs_type_id) and is_rhs_zero) and \
                    (self.is_integer(lhs_type_id) or self.is_integer(rhs_type_id)):
                self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`,'
                           f' but got `{self.type_name(rhs_type_id)}`', checked_expr.span)
                return CheckedStatement.Invalid(span)
        else:
            if lhs_type_id != rhs_type_id and rhs_type_id != unknown_type_id():
                self.error(f'Type mismatch: expected `{self.type_name(lhs_type_id)}`,'
                           f' but got `{self.type_name(rhs_type_id)}`', checked_expr.span)

        checked_var = CheckedVariable(var.name, lhs_type_id, var.is_mutable, var.span, None, Visibility.Public())

        if self.dump_type_hints and var.inlay_span:
            self.dump_type_hint(lhs_type_id, var.inlay_span)

        module = self.current_module()
        var_id = module.add_variable(checked_var)
        self.add_var_to_scope(scope_id, var.name, var_id, checked_var.definition_span)

        if checked_expr.variant == 'Function' and not self.find_function_in_scope(scope_id, var.name):
            checked_function = CheckedFunction(var.name, var.span, Visibility.Public(), checked_expr.return_type_id,
                                               None, checked_expr.params, [], checked_expr.block, checked_expr.can_throw,
                                               FunctionType.Normal(), FunctionLinkage.Internal(), scope_id,
                                               False, None, False)
            function_id = module.add_function(checked_function)
            self.add_function_to_scope(scope_id, var.name, function_id, checked_expr.span)

        return CheckedStatement.VarDecl(var_id, checked_expr, span)

    def typecheck_while(self, condition: ParsedExpression, block: ParsedBlock, scope_id: ScopeId,
                        safety_mode: SafetyMode, span, TextSpan) -> CheckedStatement:
        checked_condition = self.typecheck_expression_and_dereference_if_needed(condition, scope_id,
                                                                                safety_mode, None, span)
        if checked_condition.type() != builtin(BuiltinType.Bool()):
            self.error('Condition must be a boolean expression', condition.span)

        checked_block = self.typecheck_block(block, scope_id, safety_mode)
        if checked_block.yielded_type:
            self.error('a `while` block is not allowed to yield values', block.find_yield_span())

        return CheckedStatement.While(checked_condition, checked_block, span)

    def typecheck_try_block(self, stmt: ParsedStatement, error_name: str, error_span: TextSpan,
                            catch_block: ParsedBlock, scope_id: ScopeId, safety_mode: SafetyMode,
                            span: TextSpan) -> CheckedExpression:
        try_scope_id = self.create_scope(scope_id, True, 'try')
        checked_stmt = self.typecheck_statement(stmt, try_scope_id, safety_mode)
        error_struct_id = self.find_struct_in_prelude('Error')
        error_decl = CheckedVariable(error_name, self.get_struct(error_struct_id).type_id, False,
                                     error_span, None, Visibility.Public())

        module = self.current_module()
        error_id = module.add_variable(error_decl)

        catch_scope_id = self.create_scope(scope_id, True, 'catch')
        self.add_var_to_scope(catch_scope_id, error_name, error_id, error_span)
        checked_catch_block = self.typecheck_block(catch_block, catch_scope_id, safety_mode)

        return CheckedExpression(checked_stmt, checked_catch_block, error_name, error_span, span, void_type_id())

    def typecheck_try(self, expr: ParsedExpression, catch_block: ParsedBlock | None, catch_name: str | None,
                      scope_id: ScopeId, safety_mode: SafetyMode, span: TextSpan,
                      type_hint: TypeId | None) -> CheckedExpression:
        checked_expr = self.typecheck_expression(expr, scope_id, safety_mode, type_hint)
        error_struct_id = self.find_struct_in_prelude('Error')
        module = self.current_module()
        checked_catch_block: CheckedBlock | None = None
        expression_type_id = checked_expr.type()

        optional_struct_id = self.find_struct_in_prelude('Optional')
        optional_type = Type.GenericInstance(optional_struct_id, [expression_type_id])
        optional_type_id = self.find_or_add_type_id(optional_type)
        type_id = optional_type_id

        if catch_block:
            catch_scope_id = self.create_scope(scope_id, True, 'catch')
            if catch_name:
                error_struct_id = self.find_struct_in_prelude('Error')
                error_decl = CheckedVariable(catch_name, self.get_struct(error_struct_id).type_id, False, span,
                                             None, Visibility.Public())
                module = self.current_module()
                error_id = module.add_variable(error_decl)
                self.add_var_to_scope(catch_scope_id, catch_name, error_id, span)

            block = self.typecheck_block(catch_block, catch_scope_id, safety_mode)
            if block.control_flow.always_transfers_control() or block.yielded_type:
                if not (block.yielded_type if block.yielded_type else expression_type_id) == expression_type_id:
                    self.error_with_hint(f'Expected a value of type `{self.type_name(expression_type_id)}`, '
                                         f'but got `{self.type_name(block.yielded_type)}`', span,
                                         'Expression `catch` block must either yield the same type as the expression '
                                         'it\'s catching, or yield nothing', span)
                else:
                    type_id = block.yielded_type if block.yielded_type else expression_type_id
            checked_catch_block = block
        return CheckedExpression.Try(checked_expr, checked_catch_block, catch_name, span, type_id, expression_type_id)

    def typecheck_throw(self, expr: ParsedExpression, scope_id: ScopeId, safety_mode: SafetyMode,
                        span: TextSpan) -> CheckedStatement:
        checked_expr = self.typecheck_expression_and_dereference_if_needed(expr, scope_id, safety_mode, None, span)

        error_type_id = self.find_type_in_prelude('Error')
        if checked_expr.type() != error_type_id:
            self.error('Throw expression does not produce an error', expr.span)

        scope = self.get_scope(scope_id)
        if not scope.can_throw:
            self.error('Throw statement need to be in a try statement or a function marked as throws', expr.span)

        return CheckedStatement.Throw(checked_expr, span)

    def typecheck_loop(self, parsed_block: ParsedBlock, scope_id: ScopeId, safety_mode: SafetyMode,
                       span: TextSpan) -> CheckedStatement:
        checked_block = self.typecheck_block(parsed_block, scope_id, safety_mode)
        if checked_block.yielded_type:
            self.error('A `loop` block is not allowed to yield values', parsed_block.find_yield_span())

        return CheckedStatement.Loop(checked_block, span)

    def typecheck_defer(self, statement: ParsedStatement, scope_id: ScopeId, safety_mode: SafetyMode,
                        span: TextSpan) -> CheckedStatement:
        was_inside_defer = self.inside_defer
        self.inside_defer = True
        checked_statement = self.typecheck_statement(statement, scope_id, safety_mode)
        if checked_statement.variant == 'Block' and checked_statement.block.yielded_type:
            self.error('`yield` inside `defer` is meaningless', span)

        # In the original code this line was deferred, but since python doesnt have a defer statement, we simply do this
        # before the return statement
        self.inside_defer = was_inside_defer
        return CheckedStatement.Defer(checked_statement, span)

    def typecheck_block_statement(self, parsed_block: ParsedBlock, scope_id: ScopeId, safety_mode: SafetyMode,
                                  span: TextSpan) -> CheckedStatement:
        checked_block = self.typecheck_block(parsed_block, scope_id, safety_mode)
        if checked_block.yielded_type:
            self.error('A block used as a statement cannot yield values, as the value cannot be observed in any way',
                       parsed_block.find_yield_span())
        return CheckedStatement.Block(checked_block, span)

    def typecheck_inline_cpp(self, block: ParsedBlock, span: TextSpan, safety_mode: SafetyMode) -> CheckedStatement:
        if safety_mode.variant == 'Safe':
            self.error('Use of inline cpp block outside of unsafe block', span)

        strings: List[str] = []
        for statement in block.stmts:
            if statement.variant == 'Expression' and statement.expr.variant == 'QuotedString':
                strings.append(statement.expr.val)
            else:
                self.error('Expected block of strings', statement.expr.span)

        return CheckedStatement.InlineCpp(strings, span)

    def typecheck_return(self, expr: ParsedExpression | None, span: TextSpan, scope_id: ScopeId,
                         safety_mode: SafetyMode) -> CheckedStatement:
        if self.inside_defer:
            self.error('`return` is not allowed inside `defer`', span)
        if not expr:
            return CheckedStatement.Return(None, span)

        # TODO: ssupport returning functions from other functions
        if expr.variant == 'Function':
            self.error('Returning a function is not currently supported', span)

        type_hint: TypeId | None = None
        if self.current_function_id:
            type_hint = self.get_function(self.current_function_id).return_type_id

        checked_expr = self.typecheck_expression(expr, scope_id, safety_mode, type_hint)
        return CheckedStatement.Return(checked_expr, span)

    def typecheck_expression_and_dereference_if_needed(self, expr: ParsedExpression, scope_id: ScopeId,
                                                       safety_mode: SafetyMode, type_hint: TypeId | None,
                                                       span: TextSpan) -> CheckedExpression:
        checked_expr = self.typecheck_expression(expr, scope_id, safety_mode, type_hint)
        checked_expr_type = self.get_type(checked_expr.type())
        if checked_expr_type.variant in ['Reference', 'MutableReference']:
            checkedExpr = CheckedExpression.UnaryOp(checked_expr, CheckedUnaryOperator.Dereference(),
                                                    span, checked_expr_type.id_)

        return checked_expr

    def typecheck_indexed_struct(self, expr: ParsedExpression, field: str, scope_id: ScopeId, is_optional: bool,
                                 safety_mode: SafetyMode, span: TextSpan) -> CheckedExpression:
        checked_expr = self.typecheck_expression_and_dereference_if_needed(expr, scope_id, safety_mode, None, span)
        checked_expr_type_id = checked_expr.type()
        checked_expr_type = self.get_type(checked_expr_type_id)
        optional_struct_id = self.find_struct_in_prelude("Optional")

        if checked_expr_type.variant == 'GenericInstance':
            type_id = checked_expr_type_id
            if is_optional:
                if not checked_expr_type.id_ == optional_struct_id:
                    self.error('Optional chaining is only allowed on optional types', span)
                    return CheckedExpression.IndexedStruct(checked_expr, field, span, is_optional, unknown_type_id())
                type_id = checked_expr_type.args[0]
            type_ = self.get_type(type_id)
            if type_.variant in ['GenericInstance', 'Struct']:
                structure = self.get_struct(type_.id_)
                for member_id in structure.fields:
                    member = self.get_variable(member_id)

                    if member.name == field:
                        resolved_type_id = self.resolve_type_var(member.type_id, scope_id)
                        if is_optional:
                            resolved_type_id = self.find_or_add_type_id(
                                    Type.GenericInstance(optional_struct_id, [resolved_type_id]))
                        # FIXME: Unify with type - From jakt source
                        self.check_member_access(scope_id, structure.scope_id, member, span)
                        return CheckedExpression.IndexedStruct(checked_expr, field, span, is_optional, resolved_type_id)
                self.error(f'Unknown member of struct: {structure.name}.{field}', span)
            else:
                self.error(f'Member field access on value of non-struct type `{self.type_name(checked_expr_type_id)}`',
                           span)
        elif checked_expr_type.variant == 'Struct':
            if is_optional:
                self.error('Optional chaining if not allowed on non-optional types', span)
            structure = self.get_struct(checked_expr_type.id_)
            for member_id in structure.fields:
                member = self.get_variable(member_id)

                if member.name == field:
                    resolved_type_id = self.resolve_type_var(member.type_id, scope_id)
                    # FIXME: Unify with type - From Jakt source
                    self.check_member_access(scope_id, structure.scope_id, member, span)
                    return CheckedExpression.IndexedStruct(checked_expr, field, span, is_optional, resolved_type_id)
        else:
            self.error(f'Member field access on value of non-struct type `{self.type_name(checked_expr_type_id)}`', span)

        # FIXME: Unify with type - From Jakt source - I'm pretty sure this is unreachable?
        return CheckedExpression.IndexedStruct(checked_expr, field, span, is_optional, unknown_type_id())

    def check_member_access(self, accessor: ScopeId, accessee: ScopeId, member: CheckedVariable, span: TextSpan):
        if member.visibility.variant == 'Private':
            if not self.scope_can_access(accessor, accessee):
                self.error(f'Can\'t access variable `{member.name}` from scope '
                           f'{self.get_scope(accessor).namespace_name}, because it\'s marked private', span)
        elif member.visibility.variant == 'Restricted':
            self.check_restricted_access(accessor, 'variable', accessee, member.name, member.visibility.whitelist,
                                         member.visibility.span)

    def check_method_access(self, accessor: ScopeId, accessee: ScopeId, method: CheckedFunction, span: TextSpan):
        if method.visibility.variant == 'Private':
            if not self.scope_can_access(accessor, accessee):
                self.error(f'Can\'t access variable `{method.name}` from scope '
                           f'{self.get_scope(accessor).namespace_name}, because it\'s marked private', span)
        elif method.visibility.variant == 'Restricted':
            self.check_restricted_access(accessor, 'variable', accessee, method.name, method.visibility.whitelist,
                                         method.visibility.span)

    def check_restricted_access(self, accessor: ScopeId, accessee_kind: str, accessee: ScopeId, name: str,
                                whitelist: List[ParsedType], span: TextSpan):
        if not self.current_struct_type_id:
            self.error(f'Can\'t access {accessee_kind} `{name}` from scope '
                       f'`{self.get_scope(accessor).namespace_name}` because it\'s not in the restricted whitelist',
                       span)
            return
        own_type_id = self.current_struct_type_id
        type_ = self.get_type(own_type_id)
        if type_.variant == 'Struct':
            was_whitelisted = False
            for whitelisted_type in whitelist:
                type_id = self.typecheck_typename(whitelisted_type, accessee, None)
                # FIXME Handle typecheck failure - From Jakt source
                if type_id == own_type_id:
                    was_whitelisted = True
                    break
            if not was_whitelisted:
                self.error(f'Can\'t access {accessee_kind} ‘{name}’ from ‘{self.get_struct(type_.id_).name}’, '
                           f'because ‘{self.get_struct(type_.id_).name}’ is not in the restricted whitelist', span)
        else:
            self.error(f'Can\'t access {accessee_kind} ‘{name}’ from ‘{self.get_struct(type_.id_).name}’, '
                       f'because ‘{self.get_struct(type_.id_).name}’ is not in the restricted whitelist', span)

    def typecheck_range(self, from_: ParsedExpression, to: ParsedExpression, scope_id: ScopeId, safety_mode: SafetyMode,
                        span: TextSpan) -> Tuple[CheckedExpression, CheckedExpression, TypeId]:
        checked_from = self.typecheck_expression(from_, scope_id, safety_mode, None)
        checked_to = self.typecheck_expression(to, scope_id, safety_mode, None)

        from_type = checked_from.type()
        to_type = checked_to.type()

        # If the range starts or ends at a constant number, we try promoting the constant to the
        # type of the other end. This makes ranges like `0..array.size()` (as the 0 becomes 0uz)
        promoted_to = self.try_to_promote_constant_expr_to_type(from_type, checked_to, span)
        if promoted_to:
            checked_to = promoted_to
            to_type = checked_to.type()
        promoted_from = self.try_to_promote_constant_expr_to_type(to_type, checked_from, span)
        if promoted_from:
            checked_from = promoted_from
            from_type = checked_from.type()

        from_span = checked_from.span
        to_span = checked_to.span

        values_type_id = self.unify(from_type, from_span, to_type, from_span)
        if not values_type_id:
            self.error('Range values differ in types', span)

        range_struct_id = self.find_struct_in_prelude('Range')
        range_type = Type.GenericInstance(range_struct_id, [values_type_id if values_type_id else unknown_type_id()])
        type_id = self.find_or_add_type_id(range_type)

        return checked_from, checked_to, type_id

    def typecheck_expression(self, expr: ParsedExpression, scope_id: ScopeId, safety_mode: SafetyMode,
                             type_hint: TypeId | None) -> CheckedExpression:
        if expr.variant == 'IndexedStruct':
            return self.typecheck_indexed_struct(expr.expr, expr.field, scope_id, expr.is_optional, safety_mode, expr.span)
        elif expr.variant == 'Boolean':
            return CheckedExpression.Boolean(expr.val, expr.span)
        elif expr.variant == 'NumericConstant':
            # FIXME: Better constant support - From Jakt source
            match expr.val.variant:
                case 'I8': return CheckedExpression.NumericConstant(CheckedNumericConstant.I8(expr.val.val), expr.span, builtin(BuiltinType.I8))
                case 'I16': return CheckedExpression.NumericConstant(CheckedNumericConstant.I16(expr.val.val), expr.span, builtin(BuiltinType.I16))
                case 'I32': return CheckedExpression.NumericConstant(CheckedNumericConstant.I32(expr.val.val), expr.span, builtin(BuiltinType.I32))
                case 'I64': return CheckedExpression.NumericConstant(CheckedNumericConstant.I64(expr.val.val), expr.span, builtin(BuiltinType.I64))
                case 'U8': return CheckedExpression.NumericConstant(CheckedNumericConstant.U8(expr.val.val), expr.span, builtin(BuiltinType.U8))
                case 'U16': return CheckedExpression.NumericConstant(CheckedNumericConstant.U16(expr.val.val), expr.span, builtin(BuiltinType.U16))
                case 'U32': return CheckedExpression.NumericConstant(CheckedNumericConstant.U32(expr.val.val), expr.span, builtin(BuiltinType.U32))
                case 'U64': return CheckedExpression.NumericConstant(CheckedNumericConstant.U64(expr.val.val), expr.span, builtin(BuiltinType.U64))
                case 'USize': return CheckedExpression.NumericConstant(CheckedNumericConstant.Usize(expr.val.val), expr.span, builtin(BuiltinType.Usize))
                case 'F32': return CheckedExpression.NumericConstant(CheckedNumericConstant.F32(expr.val.val), expr.span, builtin(BuiltinType.F32))
                case 'F64': return CheckedExpression.NumericConstant(CheckedNumericConstant.F64(expr.val.val), expr.span, builtin(BuiltinType.F64))
        elif expr.variant == 'SingleQuotedString': return CheckedExpression.CharacterConstant(expr.val, expr.span)
        elif expr.variant == 'SingleQuotedByteString': return CheckedExpression.ByteConstant(expr.val, expr.span)
        elif expr.variant == 'QuotedString':
            if self.dump_try_hints:
                self.dump_try_hint(expr.span)

            self.unify_with_type(builtin(BuiltinType.String()), type_hint, expr.span)
            yield CheckedExpression.QuotedString(expr.val, expr.span)
        elif expr.variant == 'Call':
            return self.typecheck_call(expr.call, scope_id, expr.span, None, None, safety_mode, type_hint, False)
        elif expr.variant == 'MethodCall':
            checked_expr = self.typecheck_expression_and_dereference_if_needed(expr.expr, scope_id, safety_mode, None, expr.span)
            checked_expr_type_id = checked_expr.type()
            found_optional = False

            checked_expr_type = self.get_type(checked_expr_type_id)
            parent_id: CheckedExpression | None = None
            if checked_expr_type.variant == 'Struct':
                parent_id = StructOrEnumId.Struct(checked_expr_type.id_)
            elif checked_expr_type.variant == 'Enum':
                parent_id = StructOrEnumId.Enum(checked_expr_type.id_)
            elif checked_expr_type.variant == 'String':
                parent_id = StructOrEnumId.Struct(self.find_struct_in_prelude('String'))
            elif checked_expr_type.variant =='GenericInstance':
                if expr.is_optional:
                    optional_struct_id = self.find_struct_in_prelude("Optional")
                    struct_id: StructOrEnumId | None = None
                    if checked_expr_type.id_ != optional_struct_id:
                        self.error(f'Can\'t use `{self.get_struct(checked_expr_type.id_)}` as an optional type in chained call', expr.span)
                    else:
                        found_optional = True
                        args_type = self.get_type(checked_expr_type.args[0])
                        if args_type.variant in ['Struct', 'GenericInstance']:
                            struct_id = StructOrEnumId(args_type.id_)
                        elif args_type.variant in ['Enum', 'GenericEnumInstance']:
                            struct_id = StructOrEnumId.Enum(args_type.id_)
                        else:
                            self.error('Can\'t use non-struct type as an optional typer in optional chained call', expr.span)
                            found_optional = False
                            struct_id = StructOrEnumId.Struct(optional_struct_id)
                    struct_id = struct_id if struct_id else StructOrEnumId.Struct(optional_struct_id)
                else:
                    StructOrEnumId.Struct(checked_expr_type.id_)
            elif checked_expr_type.variant == 'GenericEnumInstance':
                parent_id = StructOrEnumId.Enum(checked_expr_type.id_)
            else:
                self.error(f'no methods available on value (type: {self.type_name(checked_expr_type_id)})', checked_expr.span)
                checked_args: List[Tuple[str, CheckedExpression]] = []

                parent_id = CheckedExpression.MethodCall(checked_expr, CheckedCall([], expr.call.name, checked_args, [], None, unknown_type_id(), False), expr.span, expr.is_optional, unknown_type_id())

            if expr.is_optional and not found_optional:
                self.error(f'Optional chain mismatch: expected optional chain, found {self.type_name(checked_expr_type_id)}', checked_expr.span)
            
            checked_call_expr = self.typecheck_call(expr.call, scope_id, expr.span, checked_expr, parent_id, safety_mode, type_hint, False)
            type_id = checked_call_expr.type()
            if checked_call_expr.variant == 'Call':
                result_type = checked_call_expr.call.return_type
                if expr.is_optional:
                    optional_struct_id = self.find_struct_in_prelude('Optional')
                    result_type = self.find_or_add_type_id(Type.GenericInstance(optional_struct_id, [result_type]))
                    return CheckedExpression.MethodCall(checked_expr, expr.call, expr.span, expr.is_optional, result_type)
                else:
                    self.compiler.panic('typecheck_call should return CheckedExpression.Call()')
        elif expr.variant == 'Range':
            checked_from, checked_to, type_id = self.typecheck_range(expr.from_, expr.to, scope_id, safety_mode, expr.span)
            return CheckedExpression.Range(checked_from, checked_to, expr.span, type_id)
        elif expr.variant == 'UnaryOp':
            checked_expr: CheckedExpression = CheckedExpression.Invalid(expr.span)
            if expr.op.variant == 'Dereference':
                checked_expr = self.typecheck_expression(expr.expr, scope_id, safety_mode, None)
            else:
                checked_expr = self.typecheck_expression_and_dereference_if_needed(expr.expr, scope_id, safety_mode, None, expr.span)

            checked_op: CheckedUnaryOperator
            if expr.op.variant == 'PreIncrement': return CheckedUnaryOperator.PreIncrement()
            elif expr.op.variant == 'PostIncrement': return CheckedUnaryOperator.PostIncrement()
            elif expr.op.variant == 'PreDecrement': return CheckedUnaryOperator.PreDecrement()
            elif expr.op.variant == 'PostDecrement': return CheckedUnaryOperator.PostDecrement()
            elif expr.op.variant == 'Negate': return CheckedUnaryOperator.Negate()
            elif expr.op.variant == 'Dereference': return CheckedUnaryOperator.Dereference()
            elif expr.op.variant =='Reference': return CheckedUnaryOperator.Reference()
            elif expr.op.variant =='MustableReference': return CheckedUnaryOperator.MustableReference()
            elif expr.op.variant =='LogicalNot': return CheckedUnaryOperator.LogicalNot()
            elif expr.op.variant =='BitwiseNot': return CheckedUnaryOperator.BitwiseNot()
            elif expr.op.variant =='TypeCast':
                cast = expr.op.cast
                type_id = self.typecheck_typename(cast.parsed_type(), scope_id, None)
                if cast.variant == 'Fallible':
                    optional_struct_id = self.find_struct_in_prelude('Optional')
                    optional_type = Type.GenericInstance(optional_struct_id, [type_id])
                    optional_type_id = self.find_or_add_type_id(optional_type)
                    checked_cast = CheckedTypeCast.Fallible(optional_type_id)
                elif cast.variant == 'Infallible':
                    checked_cast = CheckedTypeCast.Infallible(type_id)
                return CheckedUnaryOperator(checked_cast)
            elif expr.op.variant == 'Is':
                unchecked_name = expr.op.type_name
                old_ignore_errors = self.ignore_errors
                self.ignore_errors = True
                type_id = self.typecheck_typename(unchecked_name, scope_id, None)
                self.ignore_errors = old_ignore_errors

                operator_is = CheckedUnaryOperator.Is(type_id)
                if type_id == unknown_type_id():
                    if unchecked_name.variant == 'Name':
                        expr_type_id = checked_expr.type()
                        expr_type = self.get_type(expr_type_id)
                        if expr_type.variant == 'Enum':
                            enum_ = self.get_enum(expr_type.id_)
                            exists = False
                            for variant in enum_.variants:
                                if variant.variant in ['StructLike', 'Typed', 'Untyped']:
                                    exists = variant.name == unchecked_name.name
                                else:
                                    exists = False
                                if exists:
                                    operator_is = CheckedUnaryOperator.IsEnumVariant(variant, [], expr_type_id)
                                    break
                            if not exists:
                                self.error(f'Enum variant {unchecked_name.name} does not exist on {self.type_name(expr_type_id)}', expr.span)
                        else:
                            self.error(f'Unknown type or invalid type name: {unchecked_name.name}', expr.span)
                    else:
                        self.error('The right-hand side of an `is` operator must be a type name or enum variant', expr.span)
                return operator_is
            elif expr.op.variant =='IsEnumVariant': return self.typecheck_is_enum_variant(checked_expr, expr.op.inner, expr.op.bindings, scope_id)
        elif expr.variant == 'BinaryOp':
            checked_lhs = self.typecheck_expression_and_dereference_if_needed(expr.lhs, scope_id, safety_mode, None, expr.span)
            lhs_type = checked_lhs.type()
            checked_rhs = self.typecheck_expression_and_dereference_if_needed(expr.rhs, scope_id, safety_mode, lhs_type, expr.span)
            promoted_rhs = self.try_to_promote_constant_expr_to_type(lhs_type, checked_rhs, expr.span)
            if promoted_rhs:
                checked_rhs = promoted_rhs
            output_type = self.typecheck_binary_operation(checked_lhs, expr.op, checked_rhs, scope_id, expr.span)
            return CheckedExpression.BinaryOp(checked_lhs, expr.op, checked_rhs, expr.span, output_type)
        elif expr.variant == 'OptionalNone':
            return CheckedExpression.OptionalNone(expr.span, unknown_type_id())
        elif expr.variant == 'OptionalSome':
            checked_expr = self.typecheck_expression(expr.expr, scope_id, safety_mode, None)
            type_id = checked_expr.type()
            optional_struct_id = self.find_struct_in_prelude('Optional')
            optional_type = Type.GenericInstance(optional_struct_id, [type_id])
            optional_type_id = self.find_or_add_type_id(optional_type)
            return CheckedExpression.OptionalSome(checked_expr, expr.span, optional_type_id)
        elif expr.variant == 'Var':
            var = self.find_var_in_scope(scope_id, expr.name)
            if var:
                return CheckedExpression.Var(var, expr.span)
            else:
                self.error(f'Variable `{expr.name}` not found', expr.span)
                return CheckedExpression.Var(CheckedVariable(expr.name, type_hint if type_hint else unknown_type_id(), False, expr.span, None, Visibility.Public()), expr.span)
        elif expr.variant == 'ForcedUnwrap':
            checked_expr = self.typecheck_expression_and_dereference_if_needed(expr.expr, scope_id, safety_mode, None, expr.span)
            type = self.get_type(checked_expr.type())

            optional_struct_id = self.find_struct_in_prelude('Optional')
            weakptr_struct_id = self.find_struct_in_prelude('WeakPtr')

            type_id: TypeId | None = None
            if type.variant == 'GenericInstance':
                inner_type_id = unknown_type_id()
                if type.id_ == optional_struct_id or type.id_ == weakptr_struct_id:
                    inner_type_id = type.args[0]
                else:
                    self.error('Forced Unwrap only works on Optional', expr.span)
                type_id = inner_type_id
            else:
                self.error('Forced Unwrap only works on Optional', expr.span)
                type_id = unknown_type_id()

            return CheckedExpression.ForcedUnwrap(checked_expr, expr.span, type_id)
        elif expr.variant == 'Array':
            return self.typecheck_array(scope_id, expr.values_, expr.fill_size, expr.span, safety_mode, type_hint)
        elif expr.variant == 'Tuple':
            VOID_TYPE_ID = builtin(BuiltinType.Void())
            checked_values: List[CheckedExpression] = []
            checked_types: List[TypeId] = []

            for value in expr.values_:
                checked_value = self.typecheck_expression(value, scope_id, safety_mode, None)
                type_id = checked_value.type()
                if type_id == VOID_TYPE_ID:
                    self.error('Cannot create a tuple that contains a value of type void', value.span)
                checked_types.append(type_id)
                checked_values.append(checked_value)

            tuple_struct_id = self.find_struct_in_prelude('Tuple')
            type_id = self.find_or_add_type_id(Type.GenericInstance(tuple_struct_id, checked_types))
            # FIXME: Unify type - From Jakt source
            return CheckedExpression.Tuple(checked_values, expr.span, type_id)
        elif expr.variant == 'IndexedExpression':
            checked_base = self.typecheck_expression_and_dereference_if_needed(expr.base, scope_id, safety_mode, None, expr.span)
            checked_index = self.typecheck_expression_and_dereference_if_needed(expr.index, scope_id, safety_mode, None, expr.span)

            array_struct_id = self.find_struct_in_prelude("Array")
            array_slice_struct_id = self.find_struct_in_prelude("ArraySlice")
            dictionary_struct_id = self.find_struct_in_prelude("Dictionary")

            expr_type_id = unknown_type_id()

            result: CheckedExpression = CheckedExpression.Invalid(expr.span)
            base_type = self.get_type(checked_base.type())
            if base_type.variant == 'GenericInstance':
                if base_type.id_ == array_struct_id or base_type.id_ == array_slice_struct_id:
                    if self.is_integer(checked_index.type()):
                        result = CheckedExpression.IndexedExpression(checked_base, checked_index, expr.span, base_type.args[0])
                    else:
                        self.error('Index is not an integer', expr.span)
                elif base_type.id_ == dictionary_struct_id:
                    result = CheckedExpression.IndexedDictionary(checked_base, checked_index, expr.span, base_type.args[1])
            else:
                self.error('Index used on value that cannot be indexed', expr.span)
            return result
        elif expr.variant == 'IndexedRangeExpression':
            checked_base = self.typecheck_expression_and_dereference_if_needed(expr.base, scope_id, safety_mode, None, expr.span)
            checked_from, checked_to, type_id = self.typecheck_range(expr.from_, expr.to, scope_id, safety_mode, expr.span)

            array_struct_id = self.find_struct_in_prelude('Array')
            array_slice_struct_id = self.find_struct_in_prelude('ArraySlice')

            result: CheckedExpression = CheckedExpression.Invalid(expr.span)
            checked_base_type = self.get_type(checked_base.type())
            if checked_base_type.variant == 'GenericInstance':
                if checked_base_type.id_ == array_struct_id:
                    if self.is_integer(checked_from.type()) and self.is_integer(checked_to.type()):
                        type_id = self.find_or_add_type_id(Type.GenericInstance(array_slice_struct_id, checked_base_type.args))
                        result = CheckedExpression.IndexedRangeExpression(checked_base, checked_from, checked_to, expr.span, type_id)
                    else:
                        self.error('Range is not integers', expr.span)
            else:
                self.error('Index range used on value that cannot be indexed', expr.span)
            return result
        elif expr.variant == 'IndexedTuple':
            checked_expr = self.typecheck_expression_and_dereference_if_needed(expr.expr, scope_id, safety_mode, None, expr.span)
            tuple_struct_id = self.find_struct_in_prelude('Tuple')
            optional_struct_id = self.find_struct_in_prelude('Optional')
            expr_type_id = unknown_type_id()
            is_optional = expr.is_optional
            checked_expr_type = self.get_type(checked_expr.type())
            if checked_expr_type.variant == 'GenericInstance':
                if checked_expr_type.id_ == tuple_struct_id:
                    if is_optional:
                        self.error('Optional chaining is not allowed on a non-optional tuple type', expr.span)
                    if expr.index >= len(checked_expr_type.args):
                        self.error('Tuple index past the end of the tuple', expr.span)
                    else:
                        expr_type_id = checked_expr_type.args[expr.index]
                elif is_optional and checked_expr_type.id_ == optional_struct_id:
                    inner_type_id = checked_expr_type.args[0]
                    inner_type = self.get_type(inner_type_id)
                    if inner_type.variant == 'GenericInstance':
                        if inner_type.id_ == tuple_struct_id:
                            if expr.index >= len(inner_type.args):
                                self.error('Tuple index past the end of the tuple', expr.span)
                            else:
                                expr_type_id = self.find_or_add_type_id(optional_struct_id, [inner_type.args[expr.index]])
                    else:
                        self.error('Optional-chained tuple index used on non-tuple value', expr.span)
            elif is_optional:
                self.error('Optional-chained tuple index used on non-tuple value', expr.span)
            else:
                self.error('Tuple index used on non-tuple value', expr.span)

            return CheckedExpression.IndexedTuple(checked_expr, expr.index, expr.span, is_optional, expr_type_id)
        elif expr.variant == 'Invalid':
            return CheckedExpression.Invalid(expr.span)
        elif expr.variant == 'NamedspacedVar':
            return self.typecheck_namespaced_var_or_simple_enum_constructor_call(expr.name, expr.namespace, scope_id,
                                                                                 safety_mode, type_hint, expr.span)
        elif expr.variant == 'Match':
            return self.typecheck_match(expr.expr, expr.cases, expr.span, scope_id, safety_mode)
        elif expr.variant == 'EnumVariantArg':
            enum_variant = expr.enum_variant
            checked_expr = self.typecheck_expression_and_dereference_if_needed(expr.inner_expr, scope_id, safety_mode,
                                                                               None, expr.span)
            checked_binding = CheckedEnumVariantBinding('', '', unknown_type_id(), expr.span)
            checked_enum_variant: CheckedEnumVariant | None = None
            if enum_variant.variant in ['NamespacedName', 'Name']:
                enum_variant_type = self.get_type(checked_expr.type())
                if enum_variant_type.variant == 'Enum':
                    enum_ = self.get_enum(enum_variant_type.id_)
                    variant = self.get_enum_variant(enum_, enum_variant.name)
                    if variant:
                        checked_enum_variant = variant
                        checked_bindings = self.typecheck_enum_variant_bindings(variant, [expr.arg], expr.span)
                        # FIXME: this seems error prone...
                        checked_binding = checked_bindings[0]
                    else:
                        self.error(f'Enum variant {enum_variant.name} does not exist', enum_variant.span)
            else:
                self.error(f'Unknown type or invalid type name: {enum_variant.name}', enum_variant.span)
        elif expr.variant == 'Dictionary':
            return self.typecheck_dictionary(expr.values_, expr.span, scope_id, safety_mode, type_hint)
        elif expr.variant == 'Set':
            return self.typecheck_set(expr.values, expr.span, scope_id, safety_mode, type_hint)
        elif expr.variant == 'Function':
            return self.typecheck_lambda(expr.captures, expr.params, expr.can_throw, expr.return_type, expr.block,
                                         expr.span, scope_id, safety_mode)
        elif expr.variant == 'Try':
            return self.typecheck_try(expr.expr, expr.catch_block, expr.catch_name, scope_id, safety_mode,
                                      expr.span, type_hint)
        elif expr.variant == 'TryBlock':
            return self.typecheck_try_block(expr.stmt, expr.error_name, expr.error_span, expr.catch_block, scope_id,
                                            safety_mode, expr.span)
        elif expr.variant == 'Operator':
            self.compiler.panic(f'typecheck_expression: Encountered unknown expression variant during typechecking: {expr}')

    def typecheck_is_enum_variant(self, checked_expr: CheckedExpression, inner: ParsedType,
                                  bindings: List[EnumVariantPatternArgument], scope_id: ScopeId) -> CheckedUnaryOperator:
        old_ignore_errors = self.ignore_errors
        self.ignore_errors = True
        type_id = self.typecheck_typename(inner, scope_id, None)
        self.ignore_errors = old_ignore_errors

        checked_op = CheckedUnaryOperator.Is(type_id)
        expr_type_id = checked_expr.type()
        
        if inner.variant == 'NamespacedName':
            variant_name = inner.name
            type_ = self.get_type(expr_type_id)
            if type_.variant != 'Enum':
                self.error(f'Unknown type or invalid type name: {variant_name}', inner.span)
                return checked_op
            enum_ = self.get_enum(type_.id_)
            variant = self.get_enum_variant(enum_, variant_name)
            if not variant:
                self.error(f'Enum variant {variant_name} does not exist on {self.type_name(type_id)}', inner.span)
                return checked_op
            checked_enum_variant_bindings = self.typecheck_enum_variant_bindings(variant, bindings, inner.span)
            checked_op = CheckedUnaryOperator.IsEnumVariant(variant, checked_enum_variant_bindings, expr_type_id)
        return checked_op

    def get_enum_variant(self, enum_: CheckedEnum, variant_name: str) -> CheckedEnumVariant | None:
        for variant in enum_.variants:
            if variant.name() == variant_name:
                return variant
        return None

    def typecheck_enum_variant_bindings(self, enum_variant: CheckedEnumVariant, bindings: [EnumVariantPatternArgument],
                                        span: TextSpan) -> List[CheckedEnumVariantBinding] | None:
        if enum_variant.variant == 'Typed':
            if len(bindings) != 1:
                self.error(f'Enum variant `{enum_variant.name()}` must have exactly one argument', span)
                return None
            return [CheckedEnumVariantBinding(None, bindings[0].binding, enum_variant.type_id, span)]

        if enum_variant.variant != 'StructLike':
            return None

        checked_vars: List[CheckedVariable] = []
        checked_enum_variant_bindings: List[CheckedEnumVariantBinding] = []

        for field in enum_variant.fields:
            checked_vars.append(self.get_variable(field))

        for binding in bindings:
            for var in checked_vars:
                binding_name = binding.name if binding.name else binding.binding
                type_id = var.type_id
                if binding_name == var.name:
                    checked_enum_variant_bindings.append(CheckedEnumVariantBinding(binding.name, binding.binding, type_id, span))
                    break

        if len(checked_enum_variant_bindings) > 0:
            return checked_enum_variant_bindings

        return None

    def typecheck_lambda(self, captures: [ParsedCapture], params: [ParsedParameter], can_throw: bool,
                         return_type: ParsedType, block: ParsedBlock, span: TextSpan, scope_id: ScopeId,
                         safety_mode: SafetyMode) -> CheckedExpression:
        lambda_scope_id = self.create_scope(scope_id, can_throw, 'lambda')

        checked_captures: List[CheckedCapture] = []
        for capture in captures:
            var_in_scope = self.find_var_in_scope(scope_id, capture.name)
            if var_in_scope:
                if capture.variant == 'ByValue':
                    checked_capture = CheckedCapture.ByValue(capture.name, capture.span)
                elif capture.variant == 'ByReference':
                    checked_capture = CheckedCapture.ByReference(capture.name, capture.span)
                elif capture.variant == 'ByMutableReference':
                    checked_capture = CheckedCapture.ByMutableReference(capture.name, capture.span)
                else:
                    self.compiler.panic(f'typecheck_lambda: capture type error, expected one of `ByValue`, `ByReference` or `ByMutableReference`, got: `{capture.variant}`')
                checked_captures.append(checked_capture)
            else:
                self.error(f'Variable `{capture.name}` not found', span)

        module = self.current_module()
        checked_params: List[CheckedParameter] = []
        first = True
        for param in params:
            checked_param = self.typecheck_parameter(param, scope_id, first, None, None)
            checked_params.append(checked_param)
            var_id = module.add_variable(checked_param.variable)
            self.add_var_to_scope(lambda_scope_id, checked_param.variable.name, var_id, checked_param.variable.definition_span)

            first = False
        
        checked_block = self.typecheck_block(block, lambda_scope_id, safety_mode)
        param_type_ids: List[TypeId] = []
        for param in params:
            param_type_ids.append(self.typecheck_typename(param.variable.parsed_type, scope_id, param.variable.name))
        if return_type:
            return_type_id = self.typecheck_typename(return_type, scope_id, None)
        else:
            return_type_id = self.infer_function_return_type(checked_block)
        type_id = self.find_or_add_type_id(Type.Function(param_type_ids, can_throw, return_type_id))

        return CheckedExpression.Function(checked_captures, checked_params, can_throw, return_type_id, checked_block, span, type_id)

    def typecheck_namespaced_var_or_simple_enum_constructor_call(self, name: str, namespace_: List[str],
                                                                 scope_id: ScopeId, safety_mode: SafetyMode,
                                                                 type_hint: TypeId | None,
                                                                 span: TextSpan) -> CheckedExpression:
        scopes: List[ScopeId] = [scope_id]
        for ns in namespace_:
            scope = scopes[len[scopes] - 1]
            ns_in_scope = self.find_namespace_in_scope(scope, ns)
            enum_in_scope = self.program.find_enum_in_scope(scope, ns)
            next_scope = scope
            if ns_in_scope:
                next_scope = ns_in_scope[0]
            elif enum_in_scope:
                next_scope = self.get_enum(enum_in_scope).scope_id
            else:
                self.error(f'Namespace `{ns}` not found', span)
            scopes.append(next_scope)
        
        scope = scopes[-1]

        min_length = len(scopes) if len(scopes) <= len(namespace_) else len(namespace_)

        checked_namespaces: [CheckedNamespace] = []
        for i in range(0, min_length):
            checked_namespaces.append(CheckedNamespace(namespace_[i], scope))
            
        var = self.find_var_in_scope(scope, name)
        if var:
            return CheckedExpression.NamespacedVar(checked_namespaces, var, span)
        
        implicit_constructor_call = ParsedCall(namespace_, name, [], [])
        call_expression = self.typecheck_call(implicit_constructor_call, scope_id, span, None, None, safety_mode, type_hint, True)
        type_id = call_expression.type()
        if call_expression.variant == 'Call':
            call = call_expression.call
        else:
            self.compiler.panic('typecheck_call returned something other than a CheckedCall')

        if call.function_id:
            return CheckedExpression.Call(call, span, type_id)
        self.error(f'Variable `{name}` not found', span)
        return CheckedExpression.NamespacedVar(checked_namespaces,
                                               CheckedVariable(name, unknown_type_id(), False, span, None, Visibility.Public()),
                                               span)

    def typecheck_array(self, scope_id: ScopeId, values: [ParsedExpression], fill_size: ParsedExpression | None,
                        span: TextSpan, safety_mode: SafetyMode, type_hint: TypeId | None) -> CheckedExpression:
        if self.dump_type_hints:
            self.dump_try_hint(span)
        if not self.get_scope(scope_id).can_throw:
            message = 'Array initialization inside non-throwing_scope'
            if self.current_function_id:
                current_function = self.get_function(self.current_function_id)
                self.error_with_hint(message, span, f'Add `throws` keyword to function {current_function.name}', current_function.name_span)
            else:
                self.error(message, span)
        repeat: CheckedExpression | None = None
        if fill_size:
            fill_size_value = fill_size.val
            fill_size_checked = self.typecheck_expression_and_dereference_if_needed(fill_size_value, scope_id, safety_mode, None, span)
            fill_size_type = fill_size_checked.type()
            if not self.is_integer(fill_size_type):
                self.error(f'Type `{self.type_name(fill_size_type)}` is not convertible to an integer. Only integer values can be array fill size expressions.', fill_size_value.span)

            repeat = fill_size_checked

        array_struct_id = self.find_struct_in_prelude('Array')
        inner_type_id = unknown_type_id()
        inferred_type_span: TextSpan | None = None

        inner_hint: TypeId | None = None
        if type_hint:
            type_hint_type = self.get_type(type_hint)
            if type_hint_type.variant == 'GenericInstance' and type_hint_type.id_ == array_struct_id:
                inner_hint = type_hint_type.args[0]

        vals: List[CheckedExpression] = []
        for value in values:
            checked_expr = self.typecheck_expression(value, scope_id, safety_mode, inner_hint)
            current_value_type_id = checked_expr.type()
            if current_value_type_id == void_type_id():
                self.error('Cannot create an array with values of type void', span)

            if inner_type_id == unknown_type_id():
                inner_type_id = current_value_type_id
                inferred_type_span = value.span
            elif inner_type_id != current_value_type_id:
                self.error_with_hint(f'Type `{self.type_name(current_value_type_id)}` does not match type `{self.type_name(inner_type_id)}` of previous values in array',
                                     value.span,
                                     f'Array was inferred to store type `{self.type_name(inner_type_id)} here`',
                                     inferred_type_span)
            vals.append(checked_expr)

        if inner_type_id == unknown_type_id():
            if inner_hint:
                inner_type_id = inner_hint
            elif type_hint and type_hint != unknown_type_id():
                self.error('Cannot infer generic type for Array<T>', span)

        type_id = self.find_or_add_type_id(Type.GenericInstance(array_struct_id, [inner_type_id]))
        return CheckedExpression.Array(vals, repeat, span, type_id, inner_type_id)

    def typecheck_set(self, values: List[ParsedExpression], span: TextSpan, scope_id: ScopeId, safety_mode: SafetyMode,
                      type_hint: TypeId | None) -> CheckedExpression:
        if self.dump_type_hints:
            self.dump_try_hint(span)

        inner_type_id = unknown_type_id()
        inferred_type_span: TextSpan | None = None
        vals: List[CheckedExpression] = []

        set_struct_id = self.find_struct_in_prelude('Set')

        inner_hint: TypeId | None = None
        if type_hint:
            type_hint_type = self.get_type(type_hint)
            if type_hint_type.variant == 'GenericInstance' and type_hint_type.id_ == set_struct_id.id_:
                inner_hint = type_hint_type.args[0]
            else:
                self.compiler.panic(f'Expected Set struct, got {self.type_name(type_hint_type)}')

        for value in values:
            checked_value = self.typecheck_expression(value, scope_id, safety_mode, inner_hint)
            current_value_type_id = checked_value.type()
            if inner_type_id == unknown_type_id():
                if current_value_type_id == void_type_id():
                    self.error('Cannot create set with values of type void', span)
                inner_type_id = current_value_type_id
                inferred_type_span = value.span
            elif inner_type_id != current_value_type_id:
                self.error_with_hint(
                    f'Type `{self.type_name(current_value_type_id)}` does not match type `{self.type_name(inner_type_id)}` of previous values in array',
                    value.span,
                    f'Array was inferred to store type `{self.type_name(inner_type_id)} here`',
                    inferred_type_span)
            vals.append(checked_value)

        if inner_type_id == unknown_type_id():
            self.error('Cannot infer generic type for Set<T>', span)

        type_id = self.find_or_add_type_id(Type.GenericInstance(set_struct_id, [inner_type_id]))
        return CheckedExpression.Set(vals, span, type_id, inner_type_id)

    def typecheck_generic_arguments_method_call(self, checked_expr: CheckedExpression, call: ParsedCall,
                                                scope_id: ScopeId, span: TextSpan, is_optional: bool,
                                                safety_mode: SafetyMode) -> CheckedExpression:
        checked_args: List[Tuple[str, CheckedExpression]] = []
        for call_arg in call.args: # args: Tuple[str, TextSpan, ParsedExpression]
            name = call_arg[0]
            expr = call_arg[2]
            checked_arg_expr = self.typecheck_expression(expr, scope_id, safety_mode, None)
            checked_arg: Tuple[str, CheckedExpression] = (name, checked_arg_expr)
            checked_args.append(checked_arg)

        checked_type_args: List[TypeId] = []
        for type_arg in call.type_args:
            checked_type_args.append(self.typecheck_typename(type_arg, scope_id, None))

        return CheckedExpression.MethodCall(
                checked_expr,
                CheckedCall(
                        [],
                        call.name,
                        checked_args,
                        checked_type_args,
                        None,
                        unknown_type_id(),
                        False),
                span,
                is_optional,
                unknown_type_id())

    def typecheck_match(self, expr: ParsedExpression, cases: List[ParsedMatchCase], span: TextSpan, scope_id: ScopeId,
                        safety_mode: SafetyMode) -> CheckedExpression:
        checked_expr = self.typecheck_expression_and_dereference_if_needed(expr, scope_id, safety_mode, None, span)
        subject_type_id = checked_expr.type()
        type_to_match_on = self.get_type(subject_type_id)
        checked_cases: List[CheckedMatchCase] = []

        generic_inferences: Dict[str, str] = {}
        final_result_type: TypeId | None = None
        
        if type_to_match_on.variant == 'GenericInstance':
            enum_ = self.get_enum(type_to_match_on.id_)
            for i in range(0, len(enum_.generic_parameters)):
                generic = enum_.generic_parameters[i].to_string()
                argument_type = type_to_match_on.args[i].to_string()
                generic_inferences[generic] = argument_type

        if type_to_match_on.variant in ['Enum', 'GenericEnumInstance']:
            enum_ = self.get_enum(type_to_match_on.id_)
            seen_catch_all = False
            catch_all_span: TextSpan | None = None
            covered_variants: Set[str] = set()

            for case_ in cases:
                for pattern in case_.patterns:
                    if pattern.variant == 'EnumVariant':
                        variant_names = pattern.variant_names
                        variant_arguments = pattern.variant_arguments
                        arguments_span = pattern.arguments_span
                        if len(variant_names) == 1:
                            temp = variant_names[0]
                            variant_names = [(enum_.name, variant_names[0][1]), temp]
                        if not variant_names:
                            continue
                        if variant_names[0][0] != enum_.name:
                            self.error(f'Match case `{variant_names[0][0]}` does not match enum `{enum_.name}`', variant_names[0][1])
                            continue
                        
                        matched_variant: CheckedEnumVariant | None = None
                        variant_index: int | None = None
                        for index, variant in enumerate(enum_.variants):
                            if variant.name == variant_names[1][0]:
                                matched_variant = variant
                                variant_index = index

                        if not matched_variant:
                            self.error(f'Enum `{enum_.name}` does not contain a variant named `{variant_names[1][0]}`', case_.marker_span)
                            return CheckedExpression.Match(checked_expr, [], span, unknown_type_id(), False)

                        new_scope_id = self.create_scope(scope_id, self.get_scope(scope_id).can_throw, f'catch-enum-variant({variant_names})')
                        module = self.current_module()
                        if matched_variant.variant == 'Untyped':
                            covered_variants.add(matched_variant.name)
                            if variant_arguments:
                                self.error(f'Match case `{matched_variant.name}` cannot have arguments', arguments_span)
                        elif matched_variant.variant == 'Typed':
                            covered_variants.add(matched_variant.name)
                            if variant_arguments:
                                if len(variant_arguments) != 1:
                                    self.error(f'Match case `{matched_variant.name}` must have exactly one argument', matched_variant.span)
                                else:
                                    variant_argument = variant_arguments[0]
                                    variable_type_id = self.substitute_typevars_in_type(matched_variant.type_id, generic_inferences)
                                    var_id = module.add_variable(CheckedVariable(variant_argument.binding, variable_type_id, False, matched_variant.span, None, Visibility.Public()))
                                    self.add_var_to_scope(new_scope_id, variant_argument.binding, var_id, matched_variant.span)
                        elif matched_variant.variant == 'StructLike':
                            covered_variants.add(matched_variant.name)

                            field_variables: List[CheckedVariable] = []
                            for var_id in matched_variant.fields:
                                field_variables.append(self.program.get_variable(var_id))
                            seen_names: Set[str] = set()
                            for arg in variant_arguments:
                                if not arg.name:
                                    found_field_name = False
                                    field_names: List[str] = []
                                    for var in field_variables:
                                        field_names.append(var.name)
                                        if var.name == arg.binding:
                                            found_field_name = True
                                    if not found_field_name:
                                        unused_field_names: List[str] = []
                                        for field_name in field_names:
                                            if field_name in seen_names:
                                                continue
                                            unused_field_names.append(field_name)
                                        self.error_with_hint(f'Match case argument `{arg.binding}` for struct-like enum variant cannot be anon',
                                                             arg.span,
                                                             f'Available arguments are: {", ".join(unused_field_names)}',
                                                             arg.span)
                                        continue
                                arg_name = arg.name if arg.name else arg.binding
                                if arg_name in seen_names:
                                    self.error(f'match case argument `{arg_name}` is already defined', arg.span)
                                    continue
                                seen_names.add(arg_name)
                                matched_field_variable: CheckedVariable | None = None
                                for var in field_variables:
                                    if var.name == arg_name:
                                        matched_field_variable = var

                                if matched_field_variable:
                                    substituted_type_id = self.substitute_typevars_in_type(matched_field_variable.type_id, generic_inferences)
                                    matched_span = matched_field_variable.definition_span
                                    if self.dump_type_hints:
                                        self.dump_type_hint(matched_field_variable.type_id, arg.span)
                                    
                                    var_id = module.add_variable(CheckedVariable(arg.binding, substituted_type_id, False, matched_span, None, Visibility.Public()))
                                    self.add_var_to_scope(new_scope_id, arg.binding, var_id, span)
                                else:
                                    self.error(f'Match case argument `{arg_name}` does not exist in struct-like enum variant `{matched_variant.name}`', arg.span)
                        else:
                            self.compiler.panic(f'implement {matched_variant} match case for matched variant')

                        checked_tuple = self.typecheck_match_body(case_.body, new_scope_id, safety_mode, generic_inferences, final_result_type, case_.marker_span)
                        checked_body = checked_tuple[0]
                        final_result_type = checked_tuple[1]

                        checked_match_case = CheckedMatchCase.EnumVariant(variant_names[1][0], variant_arguments, subject_type_id, variant_index, new_scope_id, checked_body, matched_variant.span)
                        checked_cases.append(checked_match_case)
                    elif pattern.variant == 'CatchAll':
                        if seen_catch_all:
                            self.error('Multiple catch-all cases in match are not allowed', case_.marker_span)
                        else:
                            seen_catch_all = True
                            catch_all_span = case_.marker_span
                        new_scope_id = self.create_scope(scope_id, self.get_scope(scope_id).can_throw, 'catch-all')
                        checked_tuple = self.typecheck_match_body(case_.body, new_scope_id, safety_mode,
                                                                  generic_inferences, final_result_type,
                                                                  case_.marker_span)
                        checked_body = checked_tuple[0]
                        final_result_type = checked_tuple[1]

                        checked_match_case = CheckedMatchCase.CatchAll(checked_body, case_.marker_span)
                        checked_cases.append(checked_match_case)

            enum_variant_names: List[str] = []
            missing_variants: List[str] = []
            
            for variant in enum_.variants:
                enum_variant_names.append(variant.name)

            for variant in enum_variant_names:
                if variant not in covered_variants:
                    missing_variants.append(variant)

            if len(missing_variants) > 0:
                if not seen_catch_all:
                    missing_values = ', '.join(missing_variants)
                    self.error(f'match expression is not exhaustive, missing variants are: {missing_values}', span)
            else:
                if seen_catch_all:
                    self.error('all variants are covered, but an irrefutable pattern is also present', span)

        elif type_to_match_on.variant == 'Void':
            self.error('Can\'t match on `void` type', checked_expr.span)
        else:
            is_enum_match = False
            is_value_match = False
            seen_catch_all = False

            all_variants_constant = True

            for case_ in cases:
                for pattern in case_.patterns:
                    if pattern.variant == 'EnumVariant':
                        variant_names = pattern.variant_name

                        if is_value_match:
                            self.error('cannot have an enum match case in a match expression containing value matches', case_.marker_span)

                        if len(variant_names) == 0:
                            self.compiler.panic('typecheck_match: variant_names is empty.')

                        is_enum_match = True

                        # we don't know what the enum type is, but we have the type var for it, so generate a generic enum match.
                        # note that this will be fully checked when this match expression is actually instantiated.

                        new_scope_id = self.create_scope(scope_id, self.get_scope(scope_id).can_throw, f'catch-enum-variant({variant_names})')
                        checked_tuple = self.typecheck_match_body(case_.body, new_scope_id, safety_mode, generic_inferences, final_result_type, case_.marker_span)
                        checked_body = checked_tuple[0]
                        final_result_type = checked_tuple[1]

                        checked_match_case = CheckedMatchCase.EnumVariant(variant_names[-1][0], pattern.variant_arguments, subject_type_id, 0, new_scope_id, checked_body, case_.marker_span)
                        checked_cases.append(checked_match_case)
                    elif pattern.variant == 'CatchAll':
                        if seen_catch_all:
                            self.error('cannot have multiple catch-all match cases', case_.marker_span)
                        seen_catch_all = True

                        new_scope_id = self.create_scope(scope_id, self.get_scope(scope_id).can_throw, 'catch-all')
                        checked_tuple = self.typecheck_match_body(case_.body, new_scope_id, safety_mode,
                                                                  generic_inferences, final_result_type,
                                                                  case_.marker_span)
                        checked_body = checked_tuple[0]
                        final_result_type = checked_tuple[1]
                        checked_match_case = CheckedMatchCase.CatchAll(checked_body, case_.marker_span)
                        checked_cases.append(checked_match_case)
                    elif pattern.variant == 'Expression':
                        if is_enum_match:
                            self.error('Cannot have a value match in a match expression containing enum matches', case_.marker_span)
                        is_value_match = True

                        checked_expression = self.typecheck_expression(pattern.expr, scope_id, safety_mode, subject_type_id)

                        if not checked_expression.to_number_constant(self.program):
                            all_variants_constant = False

                        generic_inferences: Dict[str, str] = {}
                        self.check_types_for_compat(checked_expression.type(), subject_type_id, generic_inferences, case_.marker_span)

                        new_scope_id = self.create_scope(scope_id, self.get_scope(scope_id).can_throw, f'catch-expression({pattern.expr})')
                        checked_tuple = self.typecheck_match_body(case_.body, new_scope_id, safety_mode, generic_inferences, final_result_type, case_.marker_span)
                        checked_body = checked_tuple[0]
                        final_result_type = checked_tuple[1]

                        checked_match_case = CheckedMatchCase.Expression(checked_expression, checked_body, case_.marker_span)
                        checked_cases.append(checked_match_case)
            if is_value_match and not seen_catch_all:
                self.error('match expression is not exhaustive, a value match must contain an irrefutable `else` pattern', span)

        final_result_type = final_result_type if final_result_type else void_type_id()
        return CheckedExpression.Match(checked_expr, checked_cases, span, final_result_type, True)

    def typecheck_match_body(self, body: ParsedMatchBody, scope_id: ScopeId, safety_mode: SafetyMode,
                             generic_inferences: Dict[str, str], final_result_type: TypeId | None,
                             span: TextSpan) -> Tuple[CheckedMatchBody, TypeId | None]:
        result_type = final_result_type
        checked_match_body: CheckedMatchBody | None = None

        if body.variant == 'Block':
            block = body.block
            checked_block = self.typecheck_block(block, scope_id, safety_mode)

            if checked_block.control_flow.may_return() or checked_block.yielded_type:
                block_type_id = checked_block.yielded_type if checked_block.yielded_type else void_type_id()
                yield_span = block.find_yield_span()
                if not yield_span:
                    yield_span = span

                if result_type:
                    self.check_types_for_compat(result_type, block_type_id, generic_inferences, yield_span)
                else:
                    result_type = block_type_id

            final_body: CheckedMatchBody | None = None
            if checked_block and not checked_block.control_flow.never_returns():
                final_body = CheckedMatchBody.Expression(CheckedExpression.Block(checked_block, span, checked_block.yielded_type))
            else:
                final_body = CheckedMatchBody.Block(checked_block)

            checked_match_body = final_body
        elif body.variant == 'Expression':
            expr = body.expr
            checked_expression = self.typecheck_expression(expr, scope_id, safety_mode, result_type)
            if result_type:
                self.check_types_for_compat(result_type, checked_expression.type(), generic_inferences, span)
            else:
                result_type = checked_expression.type()
            checked_match_body = CheckedMatchBody.Expression(checked_expression)
        if not checked_match_body:
            self.compiler.panic('typecheck_match_body: checked_match_body should never be None')

        return checked_match_body, result_type

    def typecheck_dictionary(self, values: List[Tuple[ParsedExpression, ParsedExpression]], span: TextSpan,
                             scope_id: ScopeId, safety_mode: SafetyMode, type_hint: TypeId | None) -> CheckedExpression:
        self.compiler.panic('TODO: typecheck_dictionary')

    def resolve_call(self, call: ParsedCall, namespaces: List[ResolvedNamespace], span: TextSpan, scope_id: ScopeId,
                     must_be_enum_constructor: bool) -> FunctionId | None:
        self.compiler.panic('TODO: resolve_call')

    def typecheck_call(self, call: ParsedCall, caller_scope_id: ScopeId, span: TextSpan,
                       this_expr: CheckedExpression | None, parent_id: StructOrEnumId | None,
                       safety_mode: SafetyMode, type_hint: TypeId | None,
                       must_be_enum_constructor: bool) -> CheckedExpression:
        self.compiler.panic('TODO: typecheck_call')

    def resolve_default_params(self, params: List[CheckedParameter], args: List[Tuple[str, TextSpan, ParsedExpression]],
                               scope_id: ScopeId, safety_mode: SafetyMode, arg_offset: int,
                               span: TextSpan) -> List[Tuple[str, TextSpan, CheckedExpression]]:
        params_with_default_value = 0

        for param in params:
            if param.default_value:
                params_with_default_value += 1

        # NOTE: This line may need to be inverted, I forget guard semantics lol
        if not (len(args) >= len(params) - arg_offset - params_with_default_value and len(args) <= len(params) - arg_offset):
            self.error('Wrong number of arguments', span)
            return []

        consumed_arg = 0
        resolved_args: List[Tuple[str, TextSpan, CheckedExpression]] = []

        for i in range(arg_offset, len(params)):
            param = params[i]
            maybe_checked_expr: CheckedExpression | None = None
            if not param.requires_label:
                if not len(args) > consumed_arg:
                    self.error(f'Missing argument for function parameter {param.variable.name}', span)
                    continue

                name, span, expr = args[consumed_arg]
                maybe_checked_expr = self.typecheck_expression(expr, scope_id, safety_mode, param.variable.type_id)
                consumed_arg += 1
            else:
                maybe_checked_expr = param.default_value

                if len(args) > consumed_arg:
                    name, span, expr = args[consumed_arg]

                    if self.validate_argument_label(param, name, span, expr, maybe_checked_expr):
                        maybe_checked_expr = self.typecheck_expression(expr, scope_id, safety_mode, param.variable.type_id)
                        consumed_arg += 1

            if maybe_checked_expr:
                checked_arg = maybe_checked_expr
                promoted_arg = self.try_to_promote_constant_expr_to_type(param.variable.type_id, checked_arg, span)
                checked_arg = promoted_arg if promoted_arg else checked_arg
                resolved_args.append((param.variable.name, span, checked_arg))
        return resolved_args

    def resolve_type_var(self, type_var_type_id: TypeId, scope_id: ScopeId) -> TypeId:
        current_type_id = type_var_type_id

        while True:
            type_var_type = self.get_type(current_type_id)
            if type_var_type.variant == 'TypeVariable':
                maybe_found_type_id = self.find_type_in_scope(scope_id, type_var_type.type_name)
                if maybe_found_type_id:
                    if maybe_found_type_id == current_type_id:
                        return current_type_id
                    current_type_id = maybe_found_type_id
                else:
                    return current_type_id
            else:
                return current_type_id


    def validate_argument_label(self, param: CheckedParameter, label: str, span: TextSpan,
                                expr: ParsedExpression, default_value: CheckedExpression | None) -> bool:
        if label == param.variable.name:
            return True
        match expr.variant:
            case 'Var':
                if expr.name == param.variable.name:
                    return True
                if not default_value:
                    self.error(f'Wrong parameter name in argument label (expected `{param.variable.name}`, got `{expr.name}`)', span)
                return False
            case 'UnaryOp':
                if expr.op.variant in ['Reference', 'MutableReference']:
                    if expr.variant == 'Var':
                        if expr.name == param.variable.name:
                            return True
                        if not default_value:
                            self.error(
                                f'Wrong parameter name in argument label (expected `{param.variable.name}`, got `{label}`)',
                                span)
        if not default_value:
            self.error(f'Wrong parameter name in argument label (expected `{param.variable.name}`, got `{label}`)', span)
        return False
