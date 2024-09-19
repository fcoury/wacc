use std::collections::HashMap;

use crate::parser::{
    Block, BlockItem, Declaration, Exp, ForInit, FunctionDecl, Program, Statement, StorageClass,
    Type, VarDecl,
};

#[allow(unused)]
#[derive(Debug, Clone)]
struct ScopeInfo {
    info: TypeInfo,
    from_local_scope: bool,
}

#[derive(Debug, Clone)]
enum TypeInfo {
    Function(FunctionInfo),
    Variable(VariableInfo),
}

impl TypeInfo {
    pub fn is_global(&self) -> bool {
        match self {
            TypeInfo::Function(fun_info) => fun_info.attrs.is_global(),
            TypeInfo::Variable(var_info) => var_info.attrs.is_global(),
        }
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, TypeInfo::Variable(_))
    }

    pub fn is_function(&self) -> bool {
        matches!(self, TypeInfo::Function(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum CalleeScope {
    File,
    Block,
}

#[derive(Debug, Clone)]
struct FunctionInfo {
    params: Vec<VarDecl>,
    attrs: FunAttrs,
}

#[derive(Debug, Clone)]
struct VariableInfo {
    typ: Type,
    attrs: VarAttrs,
}

#[derive(Debug, Clone)]
struct SymbolMap {
    declarations: HashMap<String, ScopeInfo>,
    inside_function: bool,
}

impl SymbolMap {
    fn new() -> Self {
        Self {
            declarations: HashMap::new(),
            inside_function: false,
        }
    }

    fn get(&self, key: &str) -> Option<&TypeInfo> {
        self.declarations.get(key).map(|v| &v.info)
    }

    fn insert(&mut self, key: String, value: TypeInfo) {
        self.declarations.insert(
            key,
            ScopeInfo {
                info: value,
                from_local_scope: false,
            },
        );
    }

    // fn with_new_scope(&self, inside_function: bool) -> Self {
    //     let inner = self
    //         .declarations
    //         .iter()
    //         .map(|(k, v)| {
    //             (
    //                 k.clone(),
    //                 ScopeInfo {
    //                     info: v.info.clone(),
    //                     from_local_scope: true,
    //                 },
    //             )
    //         })
    //         .collect();
    //
    //     Self {
    //         declarations: inner,
    //         inside_function,
    //     }
    // }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InitialValue {
    Tentative,
    Initial(i32),
    NoInitializer,
}

// page 229, listing 10-24
// NOTE: this doesn't seem like the correct abstraction
// #[derive(Debug, Clone)]
// pub enum IdentifierAttrs {
//     FunAttr { defined: bool, global: bool },
//     StaticAttr { init: InitialValue, global: bool },
//     LocalAttr,
// }

#[derive(Debug, Clone)]
pub struct FunAttrs {
    defined: bool,
    global: bool,
}

impl FunAttrs {
    pub fn is_global(&self) -> bool {
        self.global
    }
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

pub fn typecheck_program(program: &Program) -> miette::Result<()> {
    let mut symbols = SymbolMap::new();
    println!("\n\ntypecheck_program:");
    for line in program.iter() {
        println!("{}", line);
    }
    for declaration in program.iter() {
        match declaration {
            Declaration::Function(fun) => {
                typecheck_function_declaration(fun, &mut symbols, CalleeScope::File)?
            }
            Declaration::Var(var) => typecheck_file_scope_variable_declaration(var, &mut symbols)?,
        }
    }
    println!();

    Ok(())
}

fn typecheck_block(symbols: &mut SymbolMap, block: &Block) -> miette::Result<()> {
    for block_item in block.iter() {
        match block_item {
            BlockItem::Declaration(declaration) => {
                // typecheck_declaration(symbols, global_fn_map, declaration)?
                match declaration {
                    Declaration::Function(fun) => {
                        typecheck_function_declaration(fun, symbols, CalleeScope::Block)?
                    }
                    Declaration::Var(var) => typecheck_local_variable_declaration(var, symbols)?,
                }
            }
            BlockItem::Statement(statement) => typecheck_statement(symbols, statement)?,
        }
    }

    Ok(())
}

// page 232, listing 10-27
fn typecheck_local_variable_declaration(
    decl: &VarDecl,
    symbols: &mut SymbolMap,
) -> miette::Result<()> {
    println!(
        "\n ** local var - {}\ntypecheck_local_variable_declaration: {decl:?}",
        decl.name
    );
    match decl.storage_class {
        Some(StorageClass::Extern) => {
            if decl.init.is_some() {
                miette::bail!(
                    "Extern variables cannot have initializers, found one for {}",
                    decl.name
                );
            }

            if let Some(old_decl) = symbols.get(&decl.name) {
                let TypeInfo::Variable(_) = old_decl else {
                    miette::bail!("Function {} redeclared as variable", decl.name);
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
            let initial_value = match decl.init {
                Some(Exp::Constant(i)) => InitialValue::Initial(i),
                None => InitialValue::Initial(0),
                _ => {
                    miette::bail!(
                        "Non-constant initializer on local static variable {}",
                        decl.name,
                    );
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
            // println!("here for {}", decl.name);
            // if let Some(old_decl) = symbols.get(&decl.name) {
            //     let TypeInfo::Variable(_) = old_decl else {
            //         miette::bail!("Function {} redeclared as variable", decl.name);
            //     };
            // }

            symbols.insert(
                decl.name.clone(),
                TypeInfo::Variable(VariableInfo {
                    typ: decl.typ.clone(),
                    attrs: VarAttrs::Local,
                }),
            );

            if let Some(exp) = &decl.init {
                typecheck_expr(symbols, exp)?;
            }
        }
    }

    Ok(())
}

// page 231, listing 10-26
fn typecheck_file_scope_variable_declaration(
    decl: &VarDecl,
    symbols: &mut SymbolMap,
) -> miette::Result<()> {
    println!(
        "\n ** var - {}\ntypecheck_file_scope_variable_declaration: {decl:?}",
        decl.name
    );
    let mut initial_value = match decl.init {
        Some(Exp::Constant(i)) => InitialValue::Initial(i),
        None => match decl.storage_class {
            Some(StorageClass::Extern) => InitialValue::NoInitializer,
            _ => InitialValue::Tentative,
        },
        _ => {
            miette::bail!("Non-constant initializer for variable {}", decl.name);
        }
    };

    let mut global = decl.storage_class != Some(StorageClass::Static);
    println!("prev global = {global}");

    // if decl.name is in symbols:
    //      old_decl = symbols.get(decl.name)
    if let Some(old_decl) = symbols.get(&decl.name) {
        let TypeInfo::Variable(old_decl) = old_decl else {
            miette::bail!("{} is not a variable", decl.name);
        };

        // if old_decl.type != Int
        if old_decl.typ != Type::Int {
            miette::bail!("{} is not an int", decl.name);
        }

        // if decl.storage_class == Extern:
        if decl.storage_class == Some(StorageClass::Extern) {
            global = old_decl.attrs.is_global();
        // else if old_decl.attrs.global != global:
        } else if old_decl.attrs.is_global() != global {
            println!("old_decl: {old_decl:?}");
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
                    initial_value = InitialValue::Initial(*i);
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

    Ok(())
}

// initial implementation:
//      page 180, listing 9-21
//      typecheck_function_declaration(decl, symbols)
// modified:
//      page 230, listing 10-26
fn typecheck_function_declaration(
    decl: &FunctionDecl,
    symbols: &mut SymbolMap,
    callee_scope: CalleeScope,
) -> miette::Result<()> {
    println!("\n ** {}\n{decl:?}", decl.name);
    // check if the function is nested
    if symbols.inside_function && decl.body.is_some() {
        miette::bail!("Nested functions are not allowed");
    }

    let mut global = decl.storage_class != Some(StorageClass::Static);
    let mut already_defined = false;

    if callee_scope == CalleeScope::Block && decl.storage_class == Some(StorageClass::Static) {
        miette::bail!("Static functions cannot be declared in block scope");
    }

    println!("  global: {global}");
    println!("  symbols: {symbols:?}");

    // if decl.name is in symbols:
    if let Some(old_decl) = symbols.get(&decl.name) {
        println!("old_decl: {:?}", old_decl);
        // if old_decl.is_variable() && old_decl.is_global() {
        //     miette::bail!("attempted to redeclare file scope variable {}", decl.name);
        // }

        if let TypeInfo::Function(FunctionInfo { params, attrs }) = old_decl {
            if params.len() != decl.params.len() {
                let plural = if params.len() == 1 { "" } else { "s" };
                miette::bail!(
                "Function {} already declared with {} parameter{plural}, new declaration found with {}",
                decl.name,
                params.len(),
                decl.params.len()
            );
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

    if let Some(body) = &decl.body {
        let old_inside_function = symbols.inside_function;
        symbols.inside_function = true;

        // let mut symbols = symbols.with_new_scope(true);
        typecheck_block(symbols, body)?;

        symbols.inside_function = old_inside_function;
    }

    Ok(())
}

fn typecheck_statement(symbols: &mut SymbolMap, statement: &Statement) -> miette::Result<()> {
    println!("\ntypecheck_statement: {statement:?}");
    match statement {
        Statement::Return(exp) => typecheck_expr(symbols, exp),
        Statement::Expression(exp) => typecheck_expr(symbols, exp),
        Statement::For {
            init,
            condition,
            post,
            body,
            label: _,
        } => {
            // let mut symbols = symbols.with_new_scope(false);
            if let Some(init) = init {
                match init {
                    // TODO: certify that we can safely ignore storage_class here
                    ForInit::Declaration(VarDecl {
                        name,
                        typ,
                        init,
                        storage_class,
                    }) => {
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
                        if let Some(init) = init {
                            typecheck_expr(symbols, init)?;
                        }
                    }
                    ForInit::Expression(Some(exp)) => typecheck_expr(symbols, exp)?,
                    ForInit::Expression(None) => (),
                }
            }
            if let Some(condition) = condition {
                typecheck_expr(symbols, condition)?;
            }
            if let Some(post) = post {
                typecheck_expr(symbols, post)?;
            }
            typecheck_statement(symbols, body)
        }
        Statement::Compound(block) => {
            // let mut symbols = symbols.with_new_scope(false);
            typecheck_block(symbols, block)
        }
        Statement::If(condition, then, otherwise) => {
            typecheck_expr(symbols, condition)?;
            typecheck_statement(symbols, then)?;
            if let Some(otherwise) = otherwise {
                typecheck_statement(symbols, otherwise)?;
            }
            Ok(())
        }
        Statement::While {
            condition,
            body,
            label: _,
        } => {
            // let mut symbols = symbols.with_new_scope(false);
            typecheck_expr(symbols, condition)?;
            typecheck_statement(symbols, body)?;

            Ok(())
        }
        Statement::DoWhile {
            body,
            condition,
            label: _,
        } => {
            // let mut symbols = symbols.with_new_scope(false);
            typecheck_expr(symbols, condition)?;
            typecheck_statement(symbols, body)?;

            Ok(())
        }
        Statement::Break(_) => Ok(()),
        Statement::Continue(_) => Ok(()),
        Statement::Null => Ok(()),
    }
}

fn typecheck_expr(symbols: &mut SymbolMap, exp: &Exp) -> miette::Result<()> {
    match exp {
        Exp::FunctionCall(name, args) => {
            if let Some(TypeInfo::Function(FunctionInfo { params, .. })) = symbols.get(name) {
                if params.len() != args.len() {
                    miette::bail!(
                        "Function {} expects {} arguments, found {}",
                        name,
                        params.len(),
                        args.len()
                    );
                }

                // TODO: check for type match here?
                //
                // for (param, arg) in params.iter().zip(args.iter()) {
                //     if param.name != arg.name {
                //         miette::bail!(
                //             "Function {} expects argument {} to be {}, found {}",
                //             name,
                //             param.name,
                //             param.name,
                //             arg.name
                //         );
                //     }
                // }
            } else {
                miette::bail!("Function {} not declared", name);
            }
        }
        Exp::Var(name) => {
            println!("\ntypecheck var: {}", name);
            match symbols.get(name) {
                None => {}
                Some(TypeInfo::Function { .. }) => {
                    miette::bail!("{} is a function, not a variable", name)
                }
                Some(TypeInfo::Variable(_)) => (),
            }
        }
        Exp::BinaryOperation(_, lhs, rhs) => {
            typecheck_expr(symbols, lhs)?;
            typecheck_expr(symbols, rhs)?;
        }
        Exp::Assignment(lhs, rhs) => {
            typecheck_expr(symbols, lhs)?;
            typecheck_expr(symbols, rhs)?;
        }
        _ => (),
    }
    Ok(())
}
