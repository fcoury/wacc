use std::collections::HashMap;

use loop_labeling::label_loops;
use type_check::type_check_program;

use crate::parser::{
    Block, BlockItem, Declaration, Exp, FunctionDecl, Program, Statement, VarDecl,
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

    pub fn run(&mut self) -> miette::Result<Program> {
        let program = self.resolve_program()?;
        let program = label_loops(&program)?;
        type_check_program(&program)?;
        Ok(program)
    }

    fn resolve_program(&mut self) -> miette::Result<Program> {
        let mut program = self.program.clone();
        let mut function_declarations = vec![];
        let mut context = Context::new();
        let mut identifier_map = IdentifierMap::new();

        for declaration in program.function_declarations {
            function_declarations.push(self.resolve_function_declaration(
                &mut context,
                &mut identifier_map,
                &declaration,
            )?);
        }

        program.function_declarations = function_declarations;
        Ok(program)
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
                BlockItem::Declaration(declaration) => items.push(BlockItem::Declaration(
                    self.resolve_declaration(context, identifier_map, declaration)?,
                )),
                BlockItem::Statement(statement) => {
                    let statement = self.resolve_statement(context, identifier_map, statement)?;
                    items.push(BlockItem::Statement(statement))
                }
            }
        }

        Ok(Block { items })
    }

    fn resolve_statement(
        &mut self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        statement: &Statement,
    ) -> miette::Result<Statement> {
        let statement = match &statement {
            Statement::Return(expr) => Statement::Return(resolve_exp(identifier_map, expr)?),
            Statement::Expression(expr) => {
                Statement::Expression(resolve_exp(identifier_map, expr)?)
            }
            Statement::Null => Statement::Null,
            Statement::If(cond, then, else_) => {
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
                Statement::If(cond, Box::new(then), else_)
            }
            Statement::Compound(block) => {
                let mut identifier_map = identifier_map.with_new_scope();
                Statement::Compound(self.resolve_block(context, &mut identifier_map, block)?)
            }
            Statement::Break(label) => Statement::Break(label.clone()),
            Statement::Continue(label) => Statement::Continue(label.clone()),
            Statement::While {
                condition,
                body,
                label,
            } => {
                let condition = resolve_exp(identifier_map, condition)?;
                let body = self.resolve_statement(context, identifier_map, body)?;
                Statement::While {
                    condition,
                    body: Box::new(body),
                    label: label.clone(),
                }
            }
            Statement::DoWhile {
                body,
                condition,
                label,
            } => {
                let body = self.resolve_statement(context, identifier_map, body)?;
                let condition = resolve_exp(identifier_map, condition)?;
                Statement::DoWhile {
                    body: Box::new(body),
                    condition,
                    label: label.clone(),
                }
            }
            Statement::For {
                init,
                condition,
                post,
                body,
                label,
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
                }
            }
        };

        Ok(statement)
    }

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
        })
    }

    fn resolve_declaration(
        &mut self,
        context: &mut Context,
        identifier_map: &mut IdentifierMap,
        declaration: &Declaration,
    ) -> miette::Result<Declaration> {
        Ok(match declaration {
            Declaration::Var(decl) => {
                Declaration::Var(resolve_var_declaration(context, identifier_map, decl)?)
            }
            Declaration::Function(decl) => Declaration::Function(
                self.resolve_function_declaration(context, identifier_map, decl)?,
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
            crate::parser::ForInit::Declaration(declaration) => {
                let declaration = Declaration::Var(VarDecl {
                    name: declaration.name.clone(),
                    init: match &declaration.init {
                        Some(exp) => Some(resolve_exp(identifier_map, exp)?),
                        None => None,
                    },
                });
                let declaration =
                    self.resolve_declaration(context, identifier_map, &declaration)?;

                match declaration {
                    Declaration::Var(var) => Ok(crate::parser::ForInit::Declaration(var)),
                    Declaration::Function(fun) => {
                        miette::bail!("Resolved function declaration for for init {}", fun.name)
                    }
                }
            }
            crate::parser::ForInit::Expression(exp) => {
                let Some(exp) = exp else {
                    return Ok(crate::parser::ForInit::Expression(None));
                };
                let exp = resolve_exp(identifier_map, exp)?;
                Ok(crate::parser::ForInit::Expression(Some(exp)))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct IdentifierMap {
    inner: HashMap<String, IdentifierInfo>,
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

    pub fn insert(&mut self, name: String, info: IdentifierInfo) {
        self.inner.insert(name, info);
    }

    pub fn overrides(&self, name: &str) -> bool {
        if let Some(info) = self.inner.get(name) {
            info.from_current_scope
        } else {
            false
        }
    }

    pub fn get(&self, name: &str) -> Option<&IdentifierInfo> {
        self.inner.get(name)
    }
}

#[derive(Debug, Clone)]
pub struct IdentifierInfo {
    name: String,
    from_current_scope: bool,
    has_linkage: bool,
}

fn resolve_param(
    context: &mut Context,
    function_name: &str,
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
        init: None,
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
        init,
    })
}

fn resolve_exp(identifier_map: &IdentifierMap, exp: &Exp) -> miette::Result<Exp> {
    match exp {
        Exp::Assignment(left, right) => {
            let left = resolve_exp(identifier_map, left.as_ref())?;
            let right = resolve_exp(identifier_map, right.as_ref())?;
            let Exp::Var(_) = left else {
                miette::bail!(
                    "Invalid assignment target. Expected variable, found {:?}",
                    left
                );
            };
            Ok(Exp::Assignment(Box::new(left), Box::new(right)))
        }
        Exp::Var(name) => {
            if let Some(info) = identifier_map.get(name) {
                Ok(Exp::Var(info.name.clone()))
            } else {
                miette::bail!("Variable {} not declared", name);
            }
        }
        // Exp::Factor(factor) => Ok(Exp::Factor(self.resolve_factor(context, factor)?)),
        Exp::Constant(_) => Ok(exp.clone()),
        Exp::Unary(op, exp) => {
            let factor = resolve_exp(identifier_map, exp.as_ref())?;
            Ok(Exp::Unary(op.clone(), Box::new(factor)))
        }
        Exp::BinaryOperation(op, left, right) => {
            let left = resolve_exp(identifier_map, left.as_ref())?;
            let right = resolve_exp(identifier_map, right.as_ref())?;
            Ok(Exp::BinaryOperation(*op, Box::new(left), Box::new(right)))
        }
        Exp::Conditional(cond, then, else_) => {
            let cond = resolve_exp(identifier_map, cond.as_ref())?;
            let then = resolve_exp(identifier_map, then.as_ref())?;
            let else_ = resolve_exp(identifier_map, else_.as_ref())?;

            Ok(Exp::Conditional(
                Box::new(cond),
                Box::new(then),
                Box::new(else_),
            ))
        }
        Exp::FunctionCall(fun_name, args) => {
            if let Some(identifier_info) = identifier_map.get(fun_name) {
                let new_args = args
                    .iter()
                    .map(|arg| resolve_exp(identifier_map, arg))
                    .collect::<miette::Result<Vec<_>>>()?;
                Ok(Exp::FunctionCall(identifier_info.name.clone(), new_args))
            } else {
                // TODO: add rastreability to items in the program
                miette::bail!("Undeclared function {fun_name}");
            }
        }
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

    fn next_var(&mut self, prefix: &str) -> String {
        let counter = self.counters.entry(prefix.to_string()).or_insert(0);
        let var_name = format!("{}_{}", prefix, counter);
        *counter += 1;
        var_name
    }

    pub fn next_label(&mut self, descr: &str) -> String {
        self.next_var(descr) // &format!("label_{descr}"))
    }
}

#[cfg(test)]
mod test {
    use crate::parser::{Block, FunctionDecl, VarDecl};

    use super::*;

    #[test]
    fn test_analysis() {
        let program = Program {
            function_declarations: vec![FunctionDecl {
                name: "main".to_string(),
                params: vec![],
                body: Some(Block {
                    items: vec![BlockItem::Declaration(Declaration::Var(VarDecl {
                        name: "x".to_string(),
                        init: None,
                    }))],
                }),
            }],
        };

        let mut analysis = Analysis::new(program);
        let program = analysis.run().unwrap();
        let body = program.function_declarations[0].body.clone().unwrap();

        assert_eq!(body.len(), 1);
        assert_eq!(
            body.items[0],
            BlockItem::Declaration(Declaration::Var(VarDecl {
                name: "x_0".to_string(),
                init: None
            }))
        );
    }

    #[test]
    fn test_dupe_var() {
        let program = Program {
            function_declarations: vec![FunctionDecl {
                name: "main".to_string(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        BlockItem::Declaration(Declaration::Var(VarDecl {
                            name: "x".to_string(),
                            init: None,
                        })),
                        BlockItem::Declaration(Declaration::Var(VarDecl {
                            name: "x".to_string(),
                            init: None,
                        })),
                    ],
                }),
            }],
        };

        let mut analysis = Analysis::new(program);
        let res = analysis.run();

        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "Variable x already declared");
    }

    fn build_program(input: &str) -> Program {
        let mut lexer = crate::lexer::Lexer::new(input);
        let tokens = lexer.run().unwrap();
        let mut parser = crate::parser::Parser::new(input, &tokens);
        parser.run().unwrap()
    }

    #[test]
    fn test_same_var_in_block() {
        let program = build_program(
            r#"
            int main(void) {
                int x;
                {
                    int x;
                }
            }
            "#,
        );

        let mut analysis = Analysis::new(program);
        let program = analysis.run().unwrap();
        let body = program.function_declarations[0].body.clone().unwrap();

        assert_eq!(body.len(), 2);
        assert_eq!(
            body.items[0],
            BlockItem::Declaration(Declaration::Var(VarDecl {
                name: "x_0".to_string(),
                init: None
            }))
        );
        assert_eq!(
            body.items[1],
            BlockItem::Statement(Statement::Compound(Block::new(vec![
                BlockItem::Declaration(Declaration::Var(VarDecl {
                    name: "x_1".to_string(),
                    init: None
                }))
            ])))
        );
    }

    // #[test]
    // fn test_label_assignment() {
    //     let while_statement = Statement::While {
    //         condition: Exp::Constant(1),
    //         body: Box::new(Statement::Expression(Exp::Constant(1))),
    //         label: None,
    //     };
    //
    //     let mut context = Context::new();
    //     let new_while = label_statement_loops(&mut context, &while_statement, None).unwrap();
    //
    //     let Statement::While { label, .. } = new_while else {
    //         panic!("Not a while");
    //     };
    //
    //     assert!(label.is_some());
    // }
}
