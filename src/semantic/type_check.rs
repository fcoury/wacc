use std::{collections::HashMap, fmt};

use miette::LabeledSpan;

use crate::parser::{
    BinaryOperator, Block, BlockItem, Const, Declaration, Exp, ForInit, FunctionDecl, Identifier,
    Program, Statement, StorageClass, Type, UnaryOperator, VarDecl,
};

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct ScopeInfo {
    pub info: TypeInfo,
    pub from_file_scope: bool,
}

impl ScopeInfo {
    pub fn as_function(&self) -> Option<&FunctionInfo> {
        match &self.info {
            TypeInfo::Function(info) => Some(info),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypeInfo {
    Function(FunctionInfo),
    Variable(VariableInfo),
}

impl TypeInfo {
    pub fn is_variable(&self) -> bool {
        matches!(self, TypeInfo::Variable(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum CalleeScope {
    File,
    Block,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub params: Vec<VarDecl>,
    pub attrs: FunAttrs,
}

#[derive(Debug, Clone)]
pub struct VariableInfo {
    pub typ: Type,
    pub attrs: VarAttrs,
}

#[derive(Debug, Clone)]
pub struct SymbolMap {
    pub declarations: HashMap<Identifier, ScopeInfo>,
    pub inside_function: bool,
    pub at_file_scope: bool,
}

impl SymbolMap {
    fn new() -> Self {
        Self {
            declarations: HashMap::new(),
            inside_function: false,
            at_file_scope: true,
        }
    }

    fn enter_block_scope(&mut self) {
        self.at_file_scope = false;
    }

    fn exit_block_scope(&mut self) {
        self.at_file_scope = true;
    }

    pub fn get(&self, key: &Identifier) -> Option<&ScopeInfo> {
        self.declarations.get(key)
    }

    fn insert(&mut self, key: Identifier, value: TypeInfo) {
        self.declarations.insert(
            key,
            ScopeInfo {
                info: value,
                from_file_scope: self.at_file_scope,
            },
        );
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Identifier, &ScopeInfo)> {
        self.declarations.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InitialValue {
    Tentative,
    Initial(StaticInit),
    NoInitializer,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StaticInit {
    IntInit(i32),
    LongInit(i64),
}

impl fmt::Display for StaticInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StaticInit::IntInit(i) => write!(f, "{}", i),
            StaticInit::LongInit(i) => write!(f, "{}", i),
        }
    }
}

impl From<Const> for StaticInit {
    fn from(c: Const) -> Self {
        match c {
            Const::Int(i) => StaticInit::IntInit(i),
            Const::Long(i) => StaticInit::LongInit(i),
        }
    }
}

// page 229, listing 10-24
// NOTE: this didn't seem like the correct abstraction
// #[derive(Debug, Clone)]
// pub enum IdentifierAttrs {
//     FunAttr { defined: bool, global: bool },
//     StaticAttr { init: InitialValue, global: bool },
//     LocalAttr,
// }

#[derive(Debug, Clone)]
pub struct FunAttrs {
    pub defined: bool,
    pub global: bool,
}

#[derive(Debug, Clone)]
pub enum VarAttrs {
    Static { init: InitialValue, global: bool },
    Local,
}

impl VarAttrs {
    pub fn is_global(&self) -> bool {
        match self {
            VarAttrs::Static { global, .. } => *global,
            VarAttrs::Local => true,
        }
    }
}

pub fn typecheck_program(source: &str, program: Program) -> miette::Result<(Program, SymbolMap)> {
    let mut symbols = SymbolMap::new();

    let mut declarations = Vec::new();
    for declaration in program.declarations.into_iter() {
        let new_declaration = match declaration {
            Declaration::Function(fun, span) => Declaration::Function(
                typecheck_function_declaration(source, fun, &mut symbols, CalleeScope::File)?,
                span,
            ),
            Declaration::Var(var, span) => Declaration::Var(
                typecheck_file_scope_variable_declaration(source, var, &mut symbols)?,
                span,
            ),
        };

        declarations.push(new_declaration);
    }

    Ok((Program { declarations }, symbols))
}

fn typecheck_block(source: &str, symbols: &mut SymbolMap, block: Block) -> miette::Result<Block> {
    symbols.enter_block_scope();

    let mut new_block_items = Vec::new();
    for block_item in block.items.into_iter() {
        let new_block_item = match block_item {
            BlockItem::Declaration(declaration, span) => match declaration {
                Declaration::Function(fun_decl, fun_span) => {
                    let fun_decl = typecheck_function_declaration(
                        source,
                        fun_decl.clone(),
                        symbols,
                        CalleeScope::Block,
                    )?;
                    BlockItem::Declaration(Declaration::Function(fun_decl, fun_span), span)
                }
                Declaration::Var(var, var_span) => {
                    let var_decl = typecheck_local_variable_declaration(source, var, symbols)?;
                    BlockItem::Declaration(Declaration::Var(var_decl, var_span), span)
                }
            },
            BlockItem::Statement(statement, span) => {
                let statement = typecheck_statement(source, symbols, statement.clone())?;
                BlockItem::Statement(statement, span)
            }
        };

        new_block_items.push(new_block_item);
    }

    symbols.exit_block_scope();
    Ok(Block::new(new_block_items, block.span))
}

// page 232, listing 10-27
fn typecheck_local_variable_declaration(
    source: &str,
    decl: VarDecl,
    symbols: &mut SymbolMap,
) -> miette::Result<VarDecl> {
    match decl.storage_class {
        Some(StorageClass::Extern) => {
            if decl.init.is_some() {
                return Err(miette::miette! {
                    labels = vec![
                        LabeledSpan::at(decl.span, "here"),
                    ],
                    "extern variables cannot have initializers, found one for {}",
                    decl.name
                }
                .with_source_code(source.to_string()));
            }

            if let Some(old_decl) = symbols.get(&decl.name) {
                let TypeInfo::Variable(_) = old_decl.info else {
                    return Err(miette::miette! {
                        labels = vec![
                            LabeledSpan::at(decl.span, "here"),
                        ],
                        "function {} redeclared as variable", decl.name
                    }
                    .with_source_code(source.to_string()));
                };
            } else {
                symbols.insert(
                    decl.name.clone(),
                    TypeInfo::Variable(VariableInfo {
                        typ: decl.typ.clone(),
                        attrs: VarAttrs::Static {
                            init: InitialValue::NoInitializer,
                            global: true,
                        },
                    }),
                );
            }
        }
        Some(StorageClass::Static) => {
            let initial_value = match &decl.init {
                Some(Exp::Constant(i, _, _)) => InitialValue::Initial(i.clone().into()),
                None => {
                    if let Some(init) = decl.typ.default_init() {
                        InitialValue::Initial(init)
                    } else {
                        InitialValue::Tentative
                    }
                }
                _ => {
                    return Err(miette::miette! {
                        labels = vec![
                            LabeledSpan::at(decl.span, "here"),
                        ],
                        "non-constant initializer on local static variable {}",
                        decl.name,
                    }
                    .with_source_code(source.to_string()));
                }
            };

            symbols.insert(
                decl.name.clone(),
                TypeInfo::Variable(VariableInfo {
                    typ: decl.typ.clone(),
                    attrs: VarAttrs::Static {
                        init: initial_value,
                        global: false,
                    },
                }),
            );
        }
        _ => {
            symbols.insert(
                decl.name.clone(),
                TypeInfo::Variable(VariableInfo {
                    typ: decl.typ.clone(),
                    attrs: VarAttrs::Local,
                }),
            );
        }
    }

    if let Some(old_decl) = symbols.get(&decl.name) {
        let TypeInfo::Variable(old_decl) = &old_decl.info else {
            return Err(miette::miette! {
                labels = vec![
                    LabeledSpan::at(decl.span, "here"),
                ],
            "{} is not a variable", decl.name
            }
            .with_source_code(source.to_string()));
        };

        if old_decl.typ != decl.typ {
            return Err(miette::miette! {
                labels = vec![
                    LabeledSpan::at(decl.span, "here"),
                ],
                "redeclaration of '{}' with a different type: '{}' vs '{}'",
                decl.name,
                old_decl.typ,
                decl.typ
            }
            .with_source_code(source.to_string()));
        }
    }

    let init = match decl.init {
        Some(exp) => Some(typecheck_exp(source, symbols, exp)?),
        None => None,
    };

    Ok(VarDecl {
        name: decl.name,
        typ: decl.typ,
        init,
        storage_class: decl.storage_class,
        span: decl.span,
    })
}

// page 231, listing 10-26
fn typecheck_file_scope_variable_declaration(
    src: &str,
    decl: VarDecl,
    symbols: &mut SymbolMap,
) -> miette::Result<VarDecl> {
    let mut initial_value = match &decl.init {
        Some(Exp::Constant(c, _, _)) => InitialValue::Initial(c.clone().into()),
        None => match decl.storage_class {
            Some(StorageClass::Extern) => InitialValue::NoInitializer,
            _ => InitialValue::Tentative,
        },
        _ => {
            miette::bail!("Non-constant initializer for variable {}", decl.name);
        }
    };

    let mut global = decl.storage_class != Some(StorageClass::Static);

    if let Some(old_decl) = symbols.get(&decl.name) {
        let TypeInfo::Variable(old_decl) = &old_decl.info else {
            miette::bail!("{} is not a variable", decl.name);
        };

        if old_decl.typ != decl.typ {
            miette::bail!(
                "redeclaration of '{}' with a different type: '{}' vs '{}'",
                decl.name,
                old_decl.typ,
                decl.typ
            );
        }

        // if decl.storage_class == Extern:
        if decl.storage_class == Some(StorageClass::Extern) {
            global = old_decl.attrs.is_global();
        // else if old_decl.attrs.global != global:
        } else if old_decl.attrs.is_global() != global {
            miette::bail!("Conflicting variable linkage declaration for {}", decl.name);
        }

        if let VarAttrs::Static { init, .. } = &old_decl.attrs {
            // if old_decl.attrs.init is a constant;
            if let InitialValue::Initial(i) = init {
                // if initial_value is a constant:
                if matches!(initial_value, InitialValue::Initial(_)) {
                    miette::bail!(
                        "Conflicting file scope variable definitions for {}",
                        decl.name
                    );
                } else {
                    // initial_value = old_decl.attrs.init
                    initial_value = InitialValue::Initial(i.clone());
                }
            }
            // else if initial_value is not a constant and old_decl.attrs.init == Tentative:
            else if !matches!(initial_value, InitialValue::Initial(_))
                && *init == InitialValue::Tentative
            {
                // initial_value = Tentative
                initial_value = InitialValue::Tentative;
            }
        }
    }

    let attrs = VarAttrs::Static {
        init: initial_value,
        global,
    };
    symbols.insert(
        decl.name.clone(),
        TypeInfo::Variable(VariableInfo {
            typ: decl.typ.clone(),
            attrs,
        }),
    );

    let init = match decl.init {
        Some(exp) => Some(typecheck_exp(src, symbols, exp)?),
        None => None,
    };

    Ok(VarDecl {
        name: decl.name,
        typ: decl.typ,
        init,
        storage_class: decl.storage_class,
        span: decl.span,
    })
}

// initial implementation:
//      page 180, listing 9-21
//      typecheck_function_declaration(decl, symbols)
// modified:
//      page 230, listing 10-26
fn typecheck_function_declaration(
    source: &str,
    decl: FunctionDecl,
    symbols: &mut SymbolMap,
    callee_scope: CalleeScope,
) -> miette::Result<FunctionDecl> {
    // check if the function is nested
    if symbols.inside_function && decl.body.is_some() {
        miette::bail!("Nested functions are not allowed");
    }

    let mut global = decl.storage_class != Some(StorageClass::Static);
    let mut already_defined = false;

    if callee_scope == CalleeScope::Block && decl.storage_class == Some(StorageClass::Static) {
        miette::bail!("Static functions cannot be declared in block scope");
    }

    // if decl.name is in symbols:
    if let Some(old_decl) = symbols.get(&decl.name) {
        if old_decl.from_file_scope && old_decl.info.is_variable() {
            miette::bail!("attempted to redeclare file scope variable {}", decl.name);
        }

        if let TypeInfo::Function(FunctionInfo { params, attrs }) = &old_decl.info {
            if params.len() != decl.params.len() {
                let plural = if params.len() == 1 { "" } else { "s" };
                miette::bail!(
                    "Function {} already declared with {} parameter{plural}, new declaration found with {}",
                    decl.name,
                    params.len(),
                    decl.params.len()
                );
            }

            for param in params.iter().zip(decl.params.iter()) {
                if param.0.typ != param.1.typ {
                    return Err(miette::miette! {
                        labels = vec![
                            LabeledSpan::at(decl.span, "here"),
                        ],
                            "Function {} already declared with parameter of type {}, new declaration found with {}",
                            decl.name,
                            param.0.typ,
                            param.1.typ
                    }
                    .with_source_code(source.to_string()));
                }
            }

            already_defined = attrs.defined;

            if attrs.global && decl.storage_class == Some(StorageClass::Static) {
                miette::bail!(
                    "Static function declaration for {} follows non-static",
                    decl.name
                );
            }

            global = attrs.global;

            if attrs.defined && decl.body.is_some() {
                miette::bail!("Function {} already declared", decl.name);
            }
        }
    }

    let attrs = FunAttrs {
        defined: already_defined || decl.body.is_some(),
        global,
    };

    // adds the function to the scope
    symbols.insert(
        decl.name.clone(),
        TypeInfo::Function(FunctionInfo {
            params: decl.params.clone(),
            attrs,
        }),
    );

    // adds all the params as vars to the scope
    for param in decl.params.iter() {
        symbols.insert(
            param.name.clone(),
            TypeInfo::Variable(VariableInfo {
                typ: param.typ.clone(),
                attrs: VarAttrs::Local,
            }),
        );
    }

    let body = match decl.body {
        Some(body) => {
            symbols.enter_block_scope();
            let old_inside_function = symbols.inside_function;
            symbols.inside_function = true;

            // let mut symbols = symbols.with_new_scope(true);
            let block = typecheck_block(source, symbols, body)?;

            symbols.inside_function = old_inside_function;
            symbols.exit_block_scope();
            Some(block)
        }
        None => None,
    };

    Ok(FunctionDecl {
        name: decl.name,
        params: decl.params,
        body,
        storage_class: decl.storage_class,
        span: decl.span,
    })
}

fn typecheck_statement(
    source: &str,
    symbols: &mut SymbolMap,
    statement: Statement,
) -> miette::Result<Statement> {
    let new_statement = match statement {
        Statement::Return(exp, span) => {
            Statement::Return(typecheck_exp(source, symbols, exp)?, span)
        }
        Statement::Expression(exp, span) => {
            Statement::Expression(typecheck_exp(source, symbols, exp)?, span)
        }
        Statement::For {
            init,
            condition,
            post,
            body,
            label,
            span,
        } => {
            // FIXME: do we need a new scope here?
            // let mut symbols = symbols.with_new_scope(false);

            let init = match init {
                // TODO: certify that we can safely ignore storage_class here
                None => None,
                Some(ForInit::Declaration(
                    VarDecl {
                        name,
                        typ,
                        init: var_init,
                        storage_class,
                        span: var_span,
                    },
                    span,
                )) => {
                    if storage_class.is_some() {
                        miette::bail!("Storage class not allowed in for loop initializer");
                    }

                    symbols.insert(
                        name.clone(),
                        TypeInfo::Variable(VariableInfo {
                            typ: typ.clone(),
                            attrs: VarAttrs::Local,
                        }),
                    );

                    let var_init_exp = match var_init {
                        Some(exp) => Some(typecheck_exp(source, symbols, exp)?),
                        None => None,
                    };

                    Some(ForInit::Declaration(
                        VarDecl {
                            name,
                            typ,
                            init: var_init_exp,
                            storage_class,
                            span: var_span,
                        },
                        span,
                    ))
                }
                Some(ForInit::Expression(Some(exp), span)) => Some(ForInit::Expression(
                    Some(typecheck_exp(source, symbols, exp)?),
                    span,
                )),
                other => other,
            };

            let condition = match condition {
                Some(exp) => Some(typecheck_exp(source, symbols, exp)?),
                None => None,
            };
            let post = match post {
                Some(exp) => Some(typecheck_exp(source, symbols, exp)?),
                None => None,
            };

            let body = typecheck_statement(source, symbols, *body)?;

            Statement::For {
                init,
                condition,
                post,
                body: Box::new(body),
                label,
                span,
            }
        }
        Statement::Compound(block, span) => {
            // let mut symbols = symbols.with_new_scope(false);
            Statement::Compound(typecheck_block(source, symbols, block)?, span)
        }
        Statement::If(condition, then, otherwise, span) => {
            let condition = typecheck_exp(source, symbols, condition)?;
            let then = typecheck_statement(source, symbols, *then)?;
            let otherwise = match otherwise {
                Some(otherwise) => Some(typecheck_statement(source, symbols, *otherwise)?),
                None => None,
            };

            Statement::If(condition, Box::new(then), otherwise.map(Box::new), span)
        }
        Statement::While {
            condition,
            body,
            label,
            span,
        } => {
            // let mut symbols = symbols.with_new_scope(false);
            let condition = typecheck_exp(source, symbols, condition)?;
            let body = typecheck_statement(source, symbols, *body)?;

            Statement::While {
                condition,
                body: Box::new(body),
                label,
                span,
            }
        }
        Statement::DoWhile {
            body,
            condition,
            label,
            span,
        } => {
            // let mut symbols = symbols.with_new_scope(false);
            let condition = typecheck_exp(source, symbols, condition)?;
            let body = typecheck_statement(source, symbols, *body)?;

            Statement::DoWhile {
                body: Box::new(body),
                condition,
                label,
                span,
            }
        }
        other => other,
    };

    Ok(new_statement)
}

fn get_commont_type(type1: Type, type2: Type) -> Type {
    if type1 == type2 {
        return type1;
    }

    Type::Long
}

fn convert_to(e: Exp, t: &Type) -> Exp {
    if e.typ() == *t {
        return e;
    }

    Exp::Cast(t.clone(), Box::new(e.clone()), t.clone(), e.span())
}

// TODO: Type Checking return Statements
// When a function returns a value, it’s implicitly converted to the function’s return type. The
// type checker needs to make this implicit conversion explicit. To type check a return statement,
// we look up the enclosing function’s return type and convert the return value to that type. This
// requires us to keep track of the name, or at least the return type, of whatever function we’re
// currently type checking. I’ll omit the pseudocode to type check return statements, since it’s
// straightforward.

fn typecheck_exp(source: &str, symbols: &SymbolMap, exp: Exp) -> miette::Result<Exp> {
    match exp {
        Exp::Var(name, _, span) => match symbols.get(&name).map(|x| &x.info) {
            None => {
                miette::bail!("Variable {} not declared", name);
            }
            Some(TypeInfo::Function { .. }) => {
                miette::bail!("{} is a function, not a variable", name)
            }
            Some(TypeInfo::Variable(variable_info)) => {
                Ok(Exp::Var(name, variable_info.typ.clone(), span))
            }
        },
        Exp::Constant(c, _, span) => {
            let new_type = match c {
                Const::Int(_) => Type::Int,
                Const::Long(_) => Type::Long,
            };

            Ok(Exp::Constant(c, new_type, span))
        }
        Exp::Cast(target_typ, inner_exp, _, span) => {
            let typed_inner = typecheck_exp(source, symbols, *inner_exp)?;
            Ok(Exp::Cast(
                target_typ.clone(),
                Box::new(typed_inner),
                target_typ.clone(),
                span,
            ))
        }
        Exp::Unary(op, inner_exp, typ, span) => {
            let typed_inner = typecheck_exp(source, symbols, *inner_exp)?;
            match op {
                UnaryOperator::Not => Ok(Exp::Unary(op, Box::new(typed_inner), Type::Int, span)),
                _ => Ok(Exp::Unary(op, Box::new(typed_inner), typ, span)),
            }
        }
        Exp::BinaryOperation(op, lhs, rhs, _, span) => {
            let exp1 = typecheck_exp(source, symbols, *lhs)?;
            let exp2 = typecheck_exp(source, symbols, *rhs)?;

            match op {
                BinaryOperator::And | BinaryOperator::Or => Ok(Exp::BinaryOperation(
                    op,
                    Box::new(exp1),
                    Box::new(exp2),
                    Type::Int,
                    span,
                )),
                _ => {
                    let common_type = get_commont_type(exp1.typ(), exp2.typ());
                    let converted_lhs = convert_to(exp1, &common_type);
                    let converted_rhs = convert_to(exp2, &common_type);

                    match op {
                        BinaryOperator::Add
                        | BinaryOperator::Subtract
                        | BinaryOperator::Multiply
                        | BinaryOperator::Divide
                        | BinaryOperator::Remainder => Ok(Exp::BinaryOperation(
                            op,
                            Box::new(converted_lhs),
                            Box::new(converted_rhs),
                            common_type,
                            span,
                        )),
                        _ => Ok(Exp::BinaryOperation(
                            op,
                            Box::new(converted_lhs),
                            Box::new(converted_rhs),
                            Type::Int,
                            span,
                        )),
                    }
                }
            }
        }
        Exp::Assignment(lhs, rhs, _, span) => {
            let typed_left = typecheck_exp(source, symbols, *lhs)?;
            let typed_right = typecheck_exp(source, symbols, *rhs)?;
            let left_type = typed_left.typ();
            let converted_right = convert_to(typed_right, &left_type);

            Ok(Exp::Assignment(
                Box::new(typed_left),
                Box::new(converted_right),
                left_type,
                span,
            ))
        }
        Exp::Conditional(cond_exp, true_exp, false_exp, _, span) => {
            typecheck_exp(source, symbols, *cond_exp.clone())?;
            let typed_true = typecheck_exp(source, symbols, *true_exp)?;
            let typed_false = typecheck_exp(source, symbols, *false_exp)?;
            let true_type = typed_true.typ();
            let converted_false = convert_to(typed_false, &true_type);

            Ok(Exp::Conditional(
                cond_exp,
                Box::new(typed_true),
                Box::new(converted_false),
                true_type,
                span,
            ))
        }
        Exp::FunctionCall(name, args, typ, span) => {
            let f_type = symbols.get(&name).map(|x| &x.info);
            match f_type {
                None => {
                    miette::bail!("Function {} not declared", name);
                }
                Some(TypeInfo::Variable { .. }) => {
                    miette::bail!("{} is a variable, not a function", name)
                }
                Some(TypeInfo::Function(FunctionInfo { params, .. })) => {
                    if params.len() != args.len() {
                        miette::bail!(
                            "Function {} expects {} arguments, found {}",
                            name,
                            params.len(),
                            args.len()
                        );
                    }

                    let mut converted_args = Vec::new();
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let typed_arg = typecheck_exp(source, symbols, arg.clone())?;
                        converted_args.push(convert_to(typed_arg, &param.typ));
                    }

                    Ok(Exp::FunctionCall(name, converted_args, typ, span))
                }
            }
        }
    }
}
