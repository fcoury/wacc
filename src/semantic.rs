use std::{collections::HashMap, sync::atomic::AtomicI32};

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
                BlockItem::Declaration(declaration) => items.push(self.resolve_declaration(
                    &mut context,
                    &mut variable_map,
                    declaration,
                )?),
                BlockItem::Statement(statement) => items.push(BlockItem::Statement(
                    self.resolve_statement(&mut context, &variable_map, statement)?,
                )),
            }
        }

        program.function_definition.body = Block { items };
        Ok(program)
    }

    fn resolve_statement(
        &mut self,
        context: &mut Context,
        variable_map: &HashMap<String, (String, bool)>,
        statement: &Statement,
    ) -> anyhow::Result<Statement> {
        let statement = match statement {
            Statement::Return(expr) => Statement::Return(resolve_exp(variable_map, expr)?),
            Statement::Expression(expr) => Statement::Expression(resolve_exp(variable_map, expr)?),
            Statement::Null => Statement::Null,
            Statement::If(cond, then, else_) => {
                let cond = resolve_exp(variable_map, cond)?;
                let then = self.resolve_statement(context, variable_map, then)?;
                let else_ = match else_ {
                    Some(else_) => Some(Box::new(self.resolve_statement(
                        context,
                        variable_map,
                        else_,
                    )?)),
                    None => None,
                };
                Statement::If(cond, Box::new(then), else_)
            }
            Statement::Compound(block) => {
                let mut variable_map = variable_map
                    .iter()
                    .map(|(k, v)| (k.clone(), (v.0.clone(), false)))
                    .collect();
                Statement::Compound(self.resolve_block(context, &mut variable_map, block)?)
            }
            Statement::Break(_) => todo!(),
            Statement::Continue(_) => todo!(),
            Statement::While(_, _, _) => todo!(),
            Statement::DoWhile(_, _, _) => todo!(),
            Statement::For(_, _, _, _, _) => todo!(),
        };

        Ok(statement)
    }

    fn resolve_block(
        &mut self,
        context: &mut Context,
        variable_map: &mut HashMap<String, (String, bool)>,
        block: &Block,
    ) -> anyhow::Result<Block> {
        let mut items = Vec::with_capacity(block.len());
        for block_item in block.iter() {
            match block_item {
                BlockItem::Declaration(declaration) => {
                    items.push(self.resolve_declaration(context, variable_map, declaration)?)
                }
                BlockItem::Statement(statement) => items.push(BlockItem::Statement(
                    self.resolve_statement(context, variable_map, statement)?,
                )),
            }
        }

        Ok(Block { items })
    }

    fn resolve_declaration(
        &mut self,
        context: &mut Context,
        variable_map: &mut HashMap<String, (String, bool)>,
        declaration: &Declaration,
    ) -> anyhow::Result<BlockItem> {
        if let Some((_, true)) = variable_map.get(&declaration.name) {
            anyhow::bail!("Variable {} already declared", declaration.name)
        }

        let unique_name = context.next_var(&declaration.name);
        variable_map.insert(declaration.name.clone(), (unique_name.clone(), true));

        let init = match &declaration.init {
            Some(exp) => Some(resolve_exp(variable_map, exp)?),
            None => None,
        };

        Ok(BlockItem::Declaration(Declaration {
            name: unique_name,
            init,
        }))
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
        let temp = self
            .next_temp
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("{}_{}", name, temp)
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
