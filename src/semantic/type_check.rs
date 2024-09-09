use std::collections::HashMap;

use crate::parser::{Block, BlockItem, Declaration, Exp, ForInit, Program, Statement, VarDecl};

#[allow(unused)]
#[derive(Debug, Clone)]
struct ScopeInfo {
    info: TypeInfo,
    from_local_scope: bool,
}

#[derive(Debug, Clone)]
enum TypeInfo {
    Function(FunctionInfo),
    Variable,
}

#[derive(Debug, Clone)]
struct FunctionInfo {
    params: Vec<VarDecl>,
    has_body: bool,
}

#[derive(Debug, Clone)]
struct TypeMap {
    declarations: HashMap<String, ScopeInfo>,
    inside_function: bool,
}

impl TypeMap {
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

    fn with_new_scope(&self, inside_function: bool) -> Self {
        let inner = self
            .declarations
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    ScopeInfo {
                        info: v.info.clone(),
                        from_local_scope: true,
                    },
                )
            })
            .collect();

        Self {
            declarations: inner,
            inside_function,
        }
    }
}

pub fn type_check_program(program: &Program) -> miette::Result<()> {
    let mut global_fn_map = HashMap::new();
    let mut type_info = TypeMap::new();
    for declaration in program.iter() {
        type_check_declaration(
            &mut type_info,
            &mut global_fn_map,
            &Declaration::Function(declaration.clone()),
        )?;
    }

    Ok(())
}

fn type_check_block(
    type_info: &mut TypeMap,
    global_fn_map: &mut HashMap<String, FunctionInfo>,
    block: &Block,
) -> miette::Result<()> {
    for block_item in block.iter() {
        match block_item {
            BlockItem::Declaration(declaration) => {
                type_check_declaration(type_info, global_fn_map, declaration)?
            }
            BlockItem::Statement(statement) => {
                type_check_statement(type_info, global_fn_map, statement)?
            }
        }
    }

    Ok(())
}

fn type_check_declaration(
    type_info: &mut TypeMap,
    global_fn_map: &mut HashMap<String, FunctionInfo>,
    declaration: &Declaration,
) -> miette::Result<()> {
    match declaration {
        Declaration::Var(decl) => {
            type_info.insert(decl.name.clone(), TypeInfo::Variable);
            if let Some(init) = &decl.init {
                type_check_expr(type_info, init)?;
            }
        }
        Declaration::Function(decl) => {
            // check if the function is nested
            if type_info.inside_function && decl.body.is_some() {
                miette::bail!("Nested functions are not allowed");
            }

            // checks if this function exists on the global map with different params
            if let Some(func_info) = global_fn_map.get(&decl.name) {
                if func_info.params.len() != decl.params.len() {
                    let plural = if func_info.params.len() == 1 { "" } else { "s" };
                    miette::bail!(
                        "Function {} already declared with {} parameter{plural}, new declaration found with {}",
                        decl.name,
                        func_info.params.len(),
                        decl.params.len()
                    );
                }
            }

            // adds the function to the global function map
            global_fn_map.insert(
                decl.name.clone(),
                FunctionInfo {
                    params: decl.params.clone(),
                    has_body: decl.body.is_some(),
                },
            );

            let prev_decl_with_body = if let Some(TypeInfo::Function(FunctionInfo {
                has_body,
                params,
            })) = type_info.get(&decl.name)
            {
                // check if the function is already declared in same scope
                if *has_body && decl.body.is_some() {
                    miette::bail!("Function {} already declared", decl.name);
                }

                // check if the function is already declared with different number of params
                if params.len() != decl.params.len() {
                    let plural = if params.len() == 1 { "" } else { "s" };
                    miette::bail!(
                            "Function {} already declared with {} parameter{plural}, new declaration found with {}",
                            decl.name,
                            params.len(),
                            decl.params.len()
                        );
                }
                *has_body
            } else {
                false
            };

            // adds the function to the scope
            type_info.insert(
                decl.name.clone(),
                TypeInfo::Function(FunctionInfo {
                    params: decl.params.clone(),
                    has_body: prev_decl_with_body || decl.body.is_some(),
                }),
            );

            // adds all the params as vars to the scope
            for param in decl.params.iter() {
                type_info.insert(param.name.clone(), TypeInfo::Variable);
            }

            if let Some(body) = &decl.body {
                let mut type_info = type_info.with_new_scope(true);
                type_check_block(&mut type_info, global_fn_map, body)?;
            }
        }
    }

    Ok(())
}

fn type_check_statement(
    type_info: &mut TypeMap,
    global_fn_map: &mut HashMap<String, FunctionInfo>,
    statement: &Statement,
) -> miette::Result<()> {
    match statement {
        Statement::Return(exp) => type_check_expr(type_info, exp),
        Statement::Expression(exp) => type_check_expr(type_info, exp),
        Statement::For {
            init,
            condition,
            post,
            body,
            label: _,
        } => {
            let mut type_info = type_info.with_new_scope(false);
            if let Some(init) = init {
                match init {
                    ForInit::Declaration(VarDecl { name, init }) => {
                        type_info.insert(name.clone(), TypeInfo::Variable);
                        if let Some(init) = init {
                            type_check_expr(&mut type_info, init)?;
                        }
                    }
                    ForInit::Expression(Some(exp)) => type_check_expr(&mut type_info, exp)?,
                    ForInit::Expression(None) => (),
                }
            }
            if let Some(condition) = condition {
                type_check_expr(&mut type_info, condition)?;
            }
            if let Some(post) = post {
                type_check_expr(&mut type_info, post)?;
            }
            type_check_statement(&mut type_info, global_fn_map, body)
        }
        Statement::Compound(block) => {
            let mut type_info = type_info.with_new_scope(false);
            type_check_block(&mut type_info, global_fn_map, block)
        }
        Statement::If(condition, then, otherwise) => {
            type_check_expr(type_info, condition)?;
            type_check_statement(type_info, global_fn_map, then)?;
            if let Some(otherwise) = otherwise {
                type_check_statement(type_info, global_fn_map, otherwise)?;
            }
            Ok(())
        }
        Statement::While {
            condition,
            body,
            label: _,
        } => {
            let mut type_info = type_info.with_new_scope(false);
            type_check_expr(&mut type_info, condition)?;
            type_check_statement(&mut type_info, global_fn_map, body)?;

            Ok(())
        }
        Statement::DoWhile {
            body,
            condition,
            label: _,
        } => {
            let mut type_info = type_info.with_new_scope(false);
            type_check_expr(&mut type_info, condition)?;
            type_check_statement(&mut type_info, global_fn_map, body)?;

            Ok(())
        }
        Statement::Break(_) => Ok(()),
        Statement::Continue(_) => Ok(()),
        Statement::Null => Ok(()),
    }
}

fn type_check_expr(type_info: &mut TypeMap, exp: &Exp) -> miette::Result<()> {
    match exp {
        Exp::FunctionCall(name, args) => {
            if let Some(TypeInfo::Function(FunctionInfo { params, .. })) = type_info.get(name) {
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
        Exp::Var(name) => match type_info.get(name) {
            None => miette::bail!("Variable {} not declared", name),
            Some(TypeInfo::Function { .. }) => {
                miette::bail!("{} is a function, not a variable", name)
            }
            Some(TypeInfo::Variable) => (),
        },
        Exp::BinaryOperation(_, lhs, rhs) => {
            type_check_expr(type_info, lhs)?;
            type_check_expr(type_info, rhs)?;
        }
        Exp::Assignment(lhs, rhs) => {
            type_check_expr(type_info, lhs)?;
            type_check_expr(type_info, rhs)?;
        }
        _ => (),
    }
    Ok(())
}
