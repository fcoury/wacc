use std::collections::HashMap;

use loop_labeling::label_loops;
use type_check::typecheck_program;
pub use type_check::{InitialValue, SymbolMap, TypeInfo, VarAttrs};

use crate::{
    lexer::Span,
    parser::{
        Block, BlockItem, Declaration, Exp, ForInit, FunctionDecl, Identifier, Program, Statement,
        StorageClass, Type, VarDecl,
    },
};

mod loop_labeling;
mod type_check;

pub struct Analysis {
    program: crate::parser::Program,
}

impl Analysis {
    pub fn new(program: Program) -> Self {
        Self { program }
    }

    pub fn run(&mut self) -> miette::Result<(SymbolMap, Program)> {
        let program = self.resolve_program()?;
        let program = label_loops(&program)?;
        let symbols = typecheck_program(&program)?;
        Ok((symbols, program))
    }

    fn resolve_program(&mut self) -> miette::Result<Program> {
        let mut program = self.program.clone();
        let mut declarations = vec![];
        let mut context = Context::new();
        let mut identifier_map = IdentifierMap::new();

        for declaration in program.declarations {
            declarations.push(match declaration {
                Declaration::Function(fun, span) => Declaration::Function(
                    self.resolve_function_declaration(&mut context, &mut identifier_map, &fun)?,
                    span,
                ),
                Declaration::Var(var, span) => Declaration::Var(
                    self.resolve_file_scope_variable_declaration(&var, &mut identifier_map),
                    span,
                ),
            })
        }

        program.declarations = declarations;
        Ok(program)
    }

    fn resolve_file_scope_variable_declaration(
        &mut self,
        decl: &VarDecl,
        identifier_map: &mut IdentifierMap,
    ) -> VarDecl {
        identifier_map.insert(
            decl.name.clone(),
            IdentifierInfo {
                name: decl.name.clone(),
                from_current_scope: true,
                has_linkage: true,
            },
        );

        decl.clone()
    }

    fn resolve_local_variable_declaration(
        &self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        decl: &VarDecl,
    ) -> miette::Result<VarDecl> {
        if let Some(prev_entry) = identifier_map.get(&decl.name) {
            if prev_entry.from_current_scope
                && !(prev_entry.has_linkage && decl.storage_class == Some(StorageClass::Extern))
            {
                // TODO: improve diagnostics when reporting this error
                miette::bail!("Conflicting local declarations for variable {}", decl.name);
            }
        }

        if decl.storage_class == Some(StorageClass::Extern) {
            identifier_map.insert(
                decl.name.clone(),
                IdentifierInfo {
                    name: decl.name.clone(),
                    from_current_scope: true,
                    has_linkage: true,
                },
            );
            Ok(decl.clone())
        } else {
            let unique_name = context.next_var(&decl.name);
            identifier_map.insert(
                decl.name.clone(),
                IdentifierInfo {
                    name: unique_name.clone(),
                    from_current_scope: true,
                    has_linkage: false,
                },
            );
            let init = match &decl.init {
                Some(exp) => Some(resolve_exp(identifier_map, exp)?),
                None => None,
            };
            Ok(VarDecl {
                name: unique_name,
                typ: decl.typ.clone(),
                init,
                storage_class: decl.storage_class.clone(),
                span: decl.span,
            })
        }
    }

    fn resolve_block(
        &mut self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        block: &Block,
    ) -> miette::Result<Block> {
        let mut items = Vec::with_capacity(block.len());
        for block_item in block.iter() {
            match block_item {
                BlockItem::Declaration(Declaration::Var(declaration, var_span), span) => items
                    .push(BlockItem::Declaration(
                        Declaration::Var(
                            self.resolve_local_variable_declaration(
                                context,
                                identifier_map,
                                declaration,
                            )?,
                            *var_span,
                        ),
                        *span,
                    )),
                BlockItem::Declaration(Declaration::Function(declaration, fun_span), span) => items
                    .push(BlockItem::Declaration(
                        Declaration::Function(
                            self.resolve_function_declaration(
                                context,
                                identifier_map,
                                declaration,
                            )?,
                            *fun_span,
                        ),
                        *span,
                    )),
                BlockItem::Statement(statement, span) => {
                    let statement = self.resolve_statement(context, identifier_map, statement)?;
                    items.push(BlockItem::Statement(statement, *span))
                }
            }
        }

        Ok(Block {
            items,
            span: block.span,
        })
    }

    fn resolve_statement(
        &mut self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        statement: &Statement,
    ) -> miette::Result<Statement> {
        let statement = match &statement {
            Statement::Return(expr, span) => {
                Statement::Return(resolve_exp(identifier_map, expr)?, *span)
            }
            Statement::Expression(expr, span) => {
                Statement::Expression(resolve_exp(identifier_map, expr)?, *span)
            }
            Statement::Null => Statement::Null,
            Statement::If(cond, then, else_, span) => {
                let cond = resolve_exp(identifier_map, cond)?;
                let then = self.resolve_statement(context, identifier_map, then)?;
                let else_ = match else_ {
                    Some(else_) => Some(Box::new(self.resolve_statement(
                        context,
                        identifier_map,
                        else_,
                    )?)),
                    None => None,
                };
                Statement::If(cond, Box::new(then), else_, *span)
            }
            Statement::Compound(block, span) => {
                let mut identifier_map = identifier_map.with_new_scope();
                Statement::Compound(
                    self.resolve_block(context, &mut identifier_map, block)?,
                    *span,
                )
            }
            Statement::Break(label, span) => Statement::Break(label.clone(), *span),
            Statement::Continue(label, span) => Statement::Continue(label.clone(), *span),
            Statement::While {
                condition,
                body,
                label,
                span,
            } => {
                let condition = resolve_exp(identifier_map, condition)?;
                let body = self.resolve_statement(context, identifier_map, body)?;
                Statement::While {
                    condition,
                    body: Box::new(body),
                    label: label.clone(),
                    span: *span,
                }
            }
            Statement::DoWhile {
                body,
                condition,
                label,
                span,
            } => {
                let body = self.resolve_statement(context, identifier_map, body)?;
                let condition = resolve_exp(identifier_map, condition)?;
                Statement::DoWhile {
                    body: Box::new(body),
                    condition,
                    label: label.clone(),
                    span: *span,
                }
            }
            Statement::For {
                init,
                condition,
                post,
                body,
                label,
                span,
            } => {
                let mut identifier_map = identifier_map.with_new_scope();
                let init = match init {
                    Some(init) => {
                        Some(self.resolve_for_init(context, &mut identifier_map, init)?)
                    }
                    None => None,
                };
                let condition = match condition {
                    Some(condition) => Some(resolve_exp(&identifier_map, condition)?),
                    None => None,
                };
                let post = match post {
                    Some(post) => Some(resolve_exp(&identifier_map, post)?),
                    None => None,
                };
                let body = self.resolve_statement(context, &mut identifier_map, body)?;
                Statement::For {
                    init,
                    condition,
                    post,
                    body: Box::new(body),
                    label: label.clone(),
                    span: *span,
                }
            }
        };

        Ok(statement)
    }

    // page 176, listing 9-19
    fn resolve_function_declaration(
        &mut self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        decl: &FunctionDecl,
    ) -> miette::Result<FunctionDecl> {
        if let Some(prev_entry) = identifier_map.get(&decl.name) {
            if prev_entry.from_current_scope && !prev_entry.has_linkage {
                miette::bail!("Function {} already declared", decl.name);
            }
        }

        identifier_map.insert(
            decl.name.clone(),
            IdentifierInfo {
                name: decl.name.clone(),
                from_current_scope: true,
                has_linkage: true,
            },
        );

        let mut new_identifier_map = identifier_map.with_new_scope();
        let new_params = decl
            .params
            .iter()
            .map(|param| resolve_param(context, &decl.name, &mut new_identifier_map, param))
            .collect::<miette::Result<Vec<_>>>()?;

        let new_body = if let Some(body) = &decl.body {
            Some(self.resolve_block(context, &mut new_identifier_map, body)?)
        } else {
            None
        };

        Ok(FunctionDecl {
            name: decl.name.clone(),
            params: new_params,
            body: new_body,
            storage_class: decl.storage_class.clone(),
        })
    }

    fn resolve_declaration(
        &mut self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        declaration: &Declaration,
    ) -> miette::Result<Declaration> {
        Ok(match declaration {
            Declaration::Var(decl, span) => Declaration::Var(
                resolve_var_declaration(context, identifier_map, decl)?,
                *span,
            ),
            Declaration::Function(decl, span) => Declaration::Function(
                self.resolve_function_declaration(context, identifier_map, decl)?,
                *span,
            ),
        })
    }

    fn resolve_for_init(
        &mut self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        init: &crate::parser::ForInit,
    ) -> miette::Result<crate::parser::ForInit> {
        match init {
            ForInit::Declaration(declaration, span) => {
                let declaration = Declaration::Var(
                    VarDecl {
                        name: declaration.name.clone(),
                        typ: Type::Int,
                        init: match &declaration.init {
                            Some(exp) => Some(resolve_exp(identifier_map, exp)?),
                            None => None,
                        },
                        storage_class: declaration.storage_class.clone(),
                        span: *span,
                    },
                    *span,
                );
                let declaration =
                    self.resolve_declaration(context, identifier_map, &declaration)?;

                match declaration {
                    Declaration::Var(var, span) => Ok(ForInit::Declaration(var, span)),
                    Declaration::Function(fun, _span) => {
                        miette::bail!("Resolved function declaration for for init {}", fun.name)
                    }
                }
            }
            ForInit::Expression(exp, span) => {
                let Some(exp) = exp else {
                    return Ok(ForInit::Expression(None, *span));
                };
                let exp = resolve_exp(identifier_map, exp)?;
                Ok(ForInit::Expression(Some(exp), *span))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct IdentifierMap {
    inner: HashMap<Identifier, IdentifierInfo>,
}

impl IdentifierMap {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn with_new_scope(&self) -> Self {
        let mut inner = HashMap::new();
        for (k, v) in self.inner.iter() {
            inner.insert(
                k.clone(),
                IdentifierInfo {
                    name: v.name.clone(),
                    from_current_scope: false,
                    has_linkage: v.has_linkage,
                },
            );
        }

        Self { inner }
    }

    pub fn insert(&mut self, name: Identifier, info: IdentifierInfo) {
        self.inner.insert(name, info);
    }

    pub fn overrides(&self, name: &Identifier) -> bool {
        if let Some(info) = self.inner.get(name) {
            info.from_current_scope
        } else {
            false
        }
    }

    pub fn get(&self, name: &Identifier) -> Option<&IdentifierInfo> {
        self.inner.get(name)
    }
}

#[derive(Debug, Clone)]
pub struct IdentifierInfo {
    name: Identifier,
    from_current_scope: bool,
    has_linkage: bool,
}

fn resolve_param(
    context: &mut Context,
    function_name: &Identifier,
    identifier_map: &mut IdentifierMap,
    param: &VarDecl,
) -> miette::Result<VarDecl> {
    if identifier_map.overrides(&param.name) {
        miette::bail!(
            "Duplicate param {} for function {}",
            param.name,
            function_name
        );
    }

    let unique_name = context.next_var(&param.name);
    identifier_map.insert(
        param.name.clone(),
        IdentifierInfo {
            name: unique_name.clone(),
            from_current_scope: true,
            has_linkage: false,
        },
    );

    Ok(VarDecl {
        name: unique_name,
        typ: Type::Int,
        init: None,
        storage_class: param.storage_class.clone(),
        span: Span::empty(),
    })
}

fn resolve_var_declaration(
    context: &mut Context,
    identifier_map: &mut IdentifierMap,
    declaration: &VarDecl,
) -> miette::Result<VarDecl> {
    if identifier_map.overrides(&declaration.name) {
        miette::bail!("Variable {} already declared", declaration.name)
    }

    let unique_name = context.next_var(&declaration.name);
    identifier_map.insert(
        declaration.name.clone(),
        IdentifierInfo {
            name: unique_name.clone(),
            from_current_scope: true,
            has_linkage: false,
        },
    );

    let init = match &declaration.init {
        Some(exp) => Some(resolve_exp(identifier_map, exp)?),
        None => None,
    };

    Ok(VarDecl {
        name: unique_name,
        typ: Type::Int,
        init,
        storage_class: declaration.storage_class.clone(),
        span: declaration.span,
    })
}

fn resolve_exp(identifier_map: &IdentifierMap, exp: &Exp) -> miette::Result<Exp> {
    match exp {
        Exp::Assignment(left, right, span) => {
            let left = resolve_exp(identifier_map, left.as_ref())?;
            let right = resolve_exp(identifier_map, right.as_ref())?;
            let Exp::Var(_, _) = left else {
                miette::bail!(
                    "Invalid assignment target. Expected variable, found {:?}",
                    left
                );
            };
            Ok(Exp::Assignment(Box::new(left), Box::new(right), *span))
        }
        Exp::Var(name, span) => {
            if let Some(info) = identifier_map.get(name) {
                Ok(Exp::Var(info.name.clone(), *span))
            } else {
                miette::bail!("Variable {} not declared", name);
            }
        }
        // Exp::Factor(factor) => Ok(Exp::Factor(self.resolve_factor(context, factor)?)),
        Exp::Constant(_, _) => Ok(exp.clone()),
        Exp::Unary(op, exp, span) => {
            let factor = resolve_exp(identifier_map, exp.as_ref())?;
            Ok(Exp::Unary(op.clone(), Box::new(factor), *span))
        }
        Exp::BinaryOperation(op, left, right, span) => {
            let left = resolve_exp(identifier_map, left.as_ref())?;
            let right = resolve_exp(identifier_map, right.as_ref())?;
            Ok(Exp::BinaryOperation(
                *op,
                Box::new(left),
                Box::new(right),
                *span,
            ))
        }
        Exp::Conditional(cond, then, else_, span) => {
            let cond = resolve_exp(identifier_map, cond.as_ref())?;
            let then = resolve_exp(identifier_map, then.as_ref())?;
            let else_ = resolve_exp(identifier_map, else_.as_ref())?;

            Ok(Exp::Conditional(
                Box::new(cond),
                Box::new(then),
                Box::new(else_),
                *span,
            ))
        }
        Exp::FunctionCall(fun_name, args, span) => {
            // page 175, listing 9-18
            if let Some(identifier_info) = identifier_map.get(fun_name) {
                let new_args = args
                    .iter()
                    .map(|arg| resolve_exp(identifier_map, arg))
                    .collect::<miette::Result<Vec<_>>>()?;
                Ok(Exp::FunctionCall(
                    identifier_info.name.clone(),
                    new_args,
                    *span,
                ))
            } else {
                // TODO: add rastreability to items in the program
                miette::bail!("Undeclared function {fun_name}");
            }
        }
        Exp::Cast(_, _exp, _span) => todo!(),
    }
}

struct Context {
    counters: HashMap<String, usize>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
        }
    }

    fn next_var(&mut self, prefix: &Identifier) -> Identifier {
        self.next_var_from(&prefix.to_string())
    }

    fn next_var_from(&mut self, prefix: &str) -> Identifier {
        let counter = self.counters.entry(prefix.to_string()).or_insert(0);
        let var_name = format!("{}_{}", prefix, counter);
        *counter += 1;
        Identifier::new(var_name)
    }

    fn next_label(&mut self, arg: &str) -> Identifier {
        self.next_var_from(arg)
    }
}

// #[cfg(test)]
// mod test {
//     use crate::parser::{Block, FunctionDecl, VarDecl};
//
//     use super::*;
//
//     #[test]
//     fn test_analysis() {
//         // let program = Program {
//         //     declarations: vec![Declaration::Function(FunctionDecl {
//         //         name: "main".to_string(),
//         //         params: vec![],
//         //         body: Some(Block {
//         //             items: vec![BlockItem::Declaration(Declaration::Var(VarDecl {
//         //                 name: "x".to_string(),
//         //                 typ: Type::Int,
//         //                 init: None,
//         //                 storage_class: None,
//         //             }))],
//         //         }),
//         //         storage_class: None,
//         //     })],
//         // };
//         let program = r#"
//         int main(void) {
//             int x;
//         }
//         "#;
//         let mut lexer = crate::lexer::Lexer::new(program);
//         let tokens = lexer.run().unwrap();
//         let mut parser = crate::parser::Parser::new(program, &tokens);
//         let program = parser.run().unwrap();
//
//         let mut analysis = Analysis::new(program);
//         let program = analysis.run().unwrap();
//         let Declaration::Function(main) = &program.declarations[0] else {
//             panic!("Not a function");
//         };
//         let body = main.body.clone().unwrap();
//
//         assert_eq!(body.len(), 1);
//         assert_eq!(
//             body.items[0],
//             BlockItem::Declaration(Declaration::Var(VarDecl {
//                 name: "x_0".to_string(),
//                 typ: Type::Int,
//                 init: None,
//                 storage_class: None,
//             }))
//         );
//     }
//
//     #[test]
//     fn test_dupe_var() {
//         let program = Program {
//             declarations: vec![Declaration::Function(FunctionDecl {
//                 name: "main".to_string(),
//                 params: vec![],
//                 body: Some(Block {
//                     items: vec![
//                         BlockItem::Declaration(Declaration::Var(VarDecl {
//                             name: "x".to_string(),
//                             typ: Type::Int,
//                             init: None,
//                             storage_class: None,
//                         })),
//                         BlockItem::Declaration(Declaration::Var(VarDecl {
//                             name: "x".to_string(),
//                             typ: Type::Int,
//                             init: None,
//                             storage_class: None,
//                         })),
//                     ],
//                 }),
//                 storage_class: None,
//             })],
//         };
//
//         let mut analysis = Analysis::new(program);
//         let res = analysis.run();
//
//         assert!(res.is_err());
//         assert_eq!(res.unwrap_err().to_string(), "Variable x already declared");
//     }
//
//     fn build_program(input: &str) -> Program {
//         let mut lexer = crate::lexer::Lexer::new(input);
//         let tokens = lexer.run().unwrap();
//         let mut parser = crate::parser::Parser::new(input, &tokens);
//         parser.run().unwrap()
//     }
//
//     #[test]
//     fn test_same_var_in_block() {
//         let program = build_program(
//             r#"
//             int main(void) {
//                 int x;
//                 {
//                     int x;
//                 }
//             }
//             "#,
//         );
//
//         let mut analysis = Analysis::new(program);
//         let program = analysis.run().unwrap();
//         let Declaration::Function(main) = &program.declarations[0] else {
//             panic!("Not a function");
//         };
//         let body = main.body.clone().unwrap();
//
//         assert_eq!(body.len(), 2);
//         assert_eq!(
//             body.items[0],
//             BlockItem::Declaration(Declaration::Var(VarDecl {
//                 name: "x_0".to_string(),
//                 typ: Type::Int,
//                 init: None,
//                 storage_class: None,
//             }))
//         );
//         assert_eq!(
//             body.items[1],
//             BlockItem::Statement(Statement::Compound(Block::new(vec![
//                 BlockItem::Declaration(Declaration::Var(VarDecl {
//                     name: "x_1".to_string(),
//                     typ: Type::Int,
//                     init: None,
//                     storage_class: None,
//                 }))
//             ])))
//         );
//     }
//
//     // #[test]
//     // fn test_label_assignment() {
//     //     let while_statement = Statement::While {
//     //         condition: Exp::Constant(1),
//     //         body: Box::new(Statement::Expression(Exp::Constant(1))),
//     //         label: None,
//     //     };
//     //
//     //     let mut context = Context::new();
//     //     let new_while = label_statement_loops(&mut context, &while_statement, None).unwrap();
//     //
//     //     let Statement::While { label, .. } = new_while else {
//     //         panic!("Not a while");
//     //     };
//     //
//     //     assert!(label.is_some());
//     // }
// }
