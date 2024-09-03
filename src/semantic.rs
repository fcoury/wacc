use std::{
    collections::HashMap,
    sync::atomic::{AtomicI32, Ordering},
};

use crate::parser::{Block, BlockItem, Declaration, Exp, Program, Statement};

pub struct Analysis {
    program: crate::parser::Program,
}

impl Analysis {
    pub fn new(program: Program) -> Self {
        Self { program }
    }

    pub fn run(&mut self) -> anyhow::Result<Program> {
        let program = self.resolve_program()?;
        Ok(program)
    }

    fn resolve_program(&mut self) -> anyhow::Result<Program> {
        let mut program = self.program.clone();
        let body = program.function_definition.body;

        let mut context = Context::new();

        let mut items = Vec::with_capacity(body.len());
        let mut variable_map = HashMap::new();
        for block_item in body.iter() {
            match block_item {
                BlockItem::Declaration(declaration) => items.push(BlockItem::Declaration(
                    resolve_declaration(&mut context, &mut variable_map, declaration)?,
                )),
                BlockItem::Statement(statement) => {
                    let statement = resolve_statement(&mut context, &mut variable_map, statement)?;
                    let statement = label_statement(&mut context, &statement, None)?;
                    items.push(BlockItem::Statement(statement))
                }
            }
        }

        program.function_definition.body = Block { items };
        Ok(program)
    }
}

fn resolve_block(
    context: &mut Context,
    variable_map: &mut HashMap<String, (String, bool)>,
    block: &Block,
) -> anyhow::Result<Block> {
    let mut items = Vec::with_capacity(block.len());
    for block_item in block.iter() {
        match block_item {
            BlockItem::Declaration(declaration) => items.push(BlockItem::Declaration(
                resolve_declaration(context, variable_map, declaration)?,
            )),
            BlockItem::Statement(statement) => {
                let statement = resolve_statement(context, variable_map, statement)?;
                items.push(BlockItem::Statement(statement))
            }
        }
    }

    Ok(Block { items })
}

fn resolve_statement(
    context: &mut Context,
    variable_map: &mut HashMap<String, (String, bool)>,
    statement: &Statement,
) -> anyhow::Result<Statement> {
    let statement = match statement {
        Statement::Return(expr) => Statement::Return(resolve_exp(variable_map, expr)?),
        Statement::Expression(expr) => Statement::Expression(resolve_exp(variable_map, expr)?),
        Statement::Null => Statement::Null,
        Statement::If(cond, then, else_) => {
            let cond = resolve_exp(variable_map, cond)?;
            let then = resolve_statement(context, variable_map, then)?;
            let else_ = match else_ {
                Some(else_) => Some(Box::new(resolve_statement(context, variable_map, else_)?)),
                None => None,
            };
            Statement::If(cond, Box::new(then), else_)
        }
        Statement::Compound(block) => {
            let mut variable_map = variable_map
                .iter()
                .map(|(k, v)| (k.clone(), (v.0.clone(), false)))
                .collect();
            Statement::Compound(resolve_block(context, &mut variable_map, block)?)
        }
        Statement::Break(label) => Statement::Break(label.clone()),
        Statement::Continue(label) => Statement::Continue(label.clone()),
        Statement::While {
            condition,
            body,
            label,
        } => {
            let condition = resolve_exp(variable_map, condition)?;
            let body = resolve_statement(context, variable_map, body)?;
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
            let body = resolve_statement(context, variable_map, body)?;
            let condition = resolve_exp(variable_map, condition)?;
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
            let mut variable_map = variable_map
                .clone()
                .iter()
                .map(|(k, v)| (k.clone(), (v.0.clone(), false)))
                .collect::<HashMap<_, _>>();
            let init = match init {
                Some(init) => Some(resolve_for_init(context, &mut variable_map, init)?),
                None => None,
            };
            let condition = match condition {
                Some(condition) => Some(resolve_exp(&variable_map, condition)?),
                None => None,
            };
            let post = match post {
                Some(post) => Some(resolve_exp(&variable_map, post)?),
                None => None,
            };
            let body = resolve_statement(context, &mut variable_map, body)?;
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

fn resolve_declaration(
    context: &mut Context,
    variable_map: &mut HashMap<String, (String, bool)>,
    declaration: &Declaration,
) -> anyhow::Result<Declaration> {
    if let Some((_, true)) = variable_map.get(&declaration.name) {
        anyhow::bail!("Variable {} already declared", declaration.name)
    }

    let unique_name = context.next_var(&declaration.name);
    variable_map.insert(declaration.name.clone(), (unique_name.clone(), true));

    let init = match &declaration.init {
        Some(exp) => Some(resolve_exp(variable_map, exp)?),
        None => None,
    };

    Ok(Declaration {
        name: unique_name,
        init,
    })
}

fn label_statement(
    context: &mut Context,
    statement: &Statement,
    current_label: Option<String>,
) -> anyhow::Result<Statement> {
    let statement = match statement {
        Statement::Break(_) => {
            if let Some(current_label) = current_label {
                return Ok(Statement::Break(Some(current_label)));
            }

            anyhow::bail!("Break statement outside of loop")
        }
        Statement::Continue(_) => {
            if let Some(current_label) = current_label {
                return Ok(Statement::Continue(Some(current_label)));
            }

            anyhow::bail!("Continue statement outside of loop")
        }
        Statement::While {
            condition,
            body,
            label,
        } => {
            let new_label = context.next_label("while");
            let body = label_statement(context, body, Some(new_label))?;
            Statement::While {
                condition: condition.clone(),
                body: Box::new(body),
                label: label.clone(),
            }
        }
        Statement::DoWhile {
            body,
            condition,
            label,
        } => {
            let new_label = context.next_label("dowhile");
            let body = label_statement(context, body, Some(new_label))?;
            Statement::DoWhile {
                body: Box::new(body),
                condition: condition.clone(),
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
            let new_label = context.next_label("for");
            let body = label_statement(context, body, Some(new_label))?;
            Statement::For {
                init: init.clone(),
                condition: condition.clone(),
                post: post.clone(),
                body: Box::new(body),
                label: label.clone(),
            }
        }
        Statement::Return(exp) => Statement::Return(exp.clone()),
        Statement::Expression(exp) => Statement::Expression(exp.clone()),
        Statement::If(exp, true_statement, false_statement) => {
            let true_statement = label_statement(context, true_statement, current_label.clone())?;
            let false_statement = match false_statement {
                Some(false_statement) => Some(Box::new(label_statement(
                    context,
                    false_statement,
                    current_label.clone(),
                )?)),
                None => None,
            };
            Statement::If(exp.clone(), Box::new(true_statement), false_statement)
        }
        Statement::Compound(block) => {
            let mut items = Vec::with_capacity(block.len());

            for block_item in block.iter() {
                match block_item {
                    BlockItem::Declaration(declaration) => {
                        items.push(BlockItem::Declaration(declaration.clone()))
                    }
                    BlockItem::Statement(statement) => {
                        let statement = label_statement(context, statement, current_label.clone())?;
                        items.push(BlockItem::Statement(statement))
                    }
                }
            }

            Statement::Compound(Block { items })
        }
        Statement::Null => Statement::Null,
    };

    Ok(statement)
}

fn resolve_for_init(
    context: &mut Context,
    variable_map: &mut HashMap<String, (String, bool)>,
    init: &crate::parser::ForInit,
) -> anyhow::Result<crate::parser::ForInit> {
    match init {
        crate::parser::ForInit::Declaration(declaration) => {
            let declaration = Declaration {
                name: declaration.name.clone(),
                init: match &declaration.init {
                    Some(exp) => Some(resolve_exp(variable_map, exp)?),
                    None => None,
                },
            };
            let declaration = resolve_declaration(context, variable_map, &declaration)?;
            Ok(crate::parser::ForInit::Declaration(declaration))
        }
        crate::parser::ForInit::Expression(exp) => {
            let Some(exp) = exp else {
                return Ok(crate::parser::ForInit::Expression(None));
            };
            let exp = resolve_exp(variable_map, exp)?;
            Ok(crate::parser::ForInit::Expression(Some(exp)))
        }
    }
}

fn resolve_exp(variable_map: &HashMap<String, (String, bool)>, exp: &Exp) -> anyhow::Result<Exp> {
    match exp {
        Exp::Assignment(left, right) => {
            let left = resolve_exp(variable_map, left.as_ref())?;
            let right = resolve_exp(variable_map, right.as_ref())?;
            let Exp::Var(_) = left else {
                anyhow::bail!(
                    "Invalid assignment target. Expected variable, found {:?}",
                    left
                );
            };
            Ok(Exp::Assignment(Box::new(left), Box::new(right)))
        }
        Exp::Var(name) => {
            if variable_map.contains_key(name) {
                let (var, _) = &variable_map[name];
                Ok(Exp::Var(var.clone()))
            } else {
                anyhow::bail!("Variable {} not declared", name);
            }
        }
        // Exp::Factor(factor) => Ok(Exp::Factor(self.resolve_factor(context, factor)?)),
        Exp::Constant(_) => Ok(exp.clone()),
        Exp::Unary(op, exp) => {
            let factor = resolve_exp(variable_map, exp.as_ref())?;
            Ok(Exp::Unary(op.clone(), Box::new(factor)))
        }
        Exp::BinaryOperation(op, left, right) => {
            let left = resolve_exp(variable_map, left.as_ref())?;
            let right = resolve_exp(variable_map, right.as_ref())?;
            Ok(Exp::BinaryOperation(*op, Box::new(left), Box::new(right)))
        }
        Exp::Conditional(cond, then, else_) => {
            let cond = resolve_exp(variable_map, cond.as_ref())?;
            let then = resolve_exp(variable_map, then.as_ref())?;
            let else_ = resolve_exp(variable_map, else_.as_ref())?;

            Ok(Exp::Conditional(
                Box::new(cond),
                Box::new(then),
                Box::new(else_),
            ))
        }
    }
}
pub struct Context {
    next_temp: AtomicI32,
}

impl Context {
    pub fn new() -> Self {
        Self {
            next_temp: AtomicI32::new(0),
        }
    }

    pub fn next_var(&self, name: &str) -> String {
        let temp = self.next_temp.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", name, temp)
    }

    pub fn next_label(&self, descr: &str) -> String {
        let id = self.next_temp.fetch_add(1, Ordering::SeqCst);
        format!("{descr}.label.{}", id)
    }
}

#[cfg(test)]
mod test {
    use crate::parser::{Block, Function};

    use super::*;

    #[test]
    fn test_analysis() {
        let program = Program {
            function_definition: Function {
                name: "main".to_string(),
                body: Block {
                    items: vec![BlockItem::Declaration(Declaration {
                        name: "x".to_string(),
                        init: None,
                    })],
                },
            },
        };

        let mut analysis = Analysis::new(program);
        let program = analysis.run().unwrap();

        assert_eq!(program.function_definition.body.len(), 1);
        assert_eq!(
            program.function_definition.body.items[0],
            BlockItem::Declaration(Declaration {
                name: "x_0".to_string(),
                init: None
            })
        );
    }

    #[test]
    fn test_dupe_var() {
        let program = Program {
            function_definition: Function {
                name: "main".to_string(),
                body: Block {
                    items: vec![
                        BlockItem::Declaration(Declaration {
                            name: "x".to_string(),
                            init: None,
                        }),
                        BlockItem::Declaration(Declaration {
                            name: "x".to_string(),
                            init: None,
                        }),
                    ],
                },
            },
        };

        let mut analysis = Analysis::new(program);
        let res = analysis.run();

        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "Variable x already declared");
    }

    fn build_program(input: &str) -> Program {
        let mut lexer = crate::lexer::Lexer::new(input);
        let tokens = lexer.run().unwrap();
        let mut parser = crate::parser::Parser::new(&tokens);
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

        assert_eq!(program.function_definition.body.len(), 2);
        assert_eq!(
            program.function_definition.body.items[0],
            BlockItem::Declaration(Declaration {
                name: "x_0".to_string(),
                init: None
            })
        );
        assert_eq!(
            program.function_definition.body.items[1],
            BlockItem::Statement(Statement::Compound(Block::new(vec![
                BlockItem::Declaration(Declaration {
                    name: "x_1".to_string(),
                    init: None
                })
            ])))
        );
    }
}
