use std::{collections::HashMap, sync::atomic::AtomicI32};

use crate::parser::{BlockItem, Declaration, Exp, Program, Statement};

pub struct Analysis {
    program: crate::parser::Program,
    variable_map: HashMap<String, String>,
}

impl Analysis {
    pub fn new(program: Program) -> Self {
        Self {
            program,
            variable_map: HashMap::new(),
        }
    }

    pub fn run(&mut self) -> anyhow::Result<Program> {
        let program = self.resolve_program()?;
        Ok(program)
    }

    fn resolve_program(&mut self) -> anyhow::Result<Program> {
        let mut program = self.program.clone();
        let items = program.function_definition.body;

        let mut context = Context::new();

        let mut block_items = Vec::with_capacity(items.len());
        for block_item in items.iter() {
            match block_item {
                BlockItem::Declaration(declaration) => {
                    block_items.push(self.resolve_declaration(&mut context, declaration)?)
                }
                BlockItem::Statement(statement) => block_items.push(BlockItem::Statement(
                    self.resolve_statement(&mut context, statement)?,
                )),
            }
        }

        program.function_definition.body = block_items;
        Ok(program)
    }

    fn resolve_statement(
        &mut self,
        context: &mut Context,
        statement: &Statement,
    ) -> anyhow::Result<Statement> {
        let statement = match statement {
            Statement::Return(expr) => Statement::Return(self.resolve_exp(context, expr)?),
            Statement::Expression(expr) => Statement::Expression(self.resolve_exp(context, expr)?),
            Statement::Null => Statement::Null,
            Statement::If(cond, then, else_) => {
                let cond = self.resolve_exp(context, cond)?;
                let then = self.resolve_statement(context, then)?;
                let else_ = match else_ {
                    Some(else_) => Some(Box::new(self.resolve_statement(context, else_)?)),
                    None => None,
                };
                Statement::If(cond, Box::new(then), else_)
            }
        };

        Ok(statement)
    }

    fn resolve_exp(&mut self, context: &mut Context, exp: &Exp) -> anyhow::Result<Exp> {
        match exp {
            Exp::Assignment(left, right) => {
                let left = self.resolve_exp(context, left.as_ref())?;
                let right = self.resolve_exp(context, right.as_ref())?;
                let Exp::Var(_) = left else {
                    anyhow::bail!(
                        "Invalid assignment target. Expected variable, found {:?}",
                        left
                    );
                };
                Ok(Exp::Assignment(Box::new(left), Box::new(right)))
            }
            Exp::Var(name) => {
                if self.variable_map.contains_key(name) {
                    Ok(Exp::Var(self.variable_map[name].clone()))
                } else {
                    anyhow::bail!("Variable {} not declared", name);
                }
            }
            // Exp::Factor(factor) => Ok(Exp::Factor(self.resolve_factor(context, factor)?)),
            Exp::Constant(_) => Ok(exp.clone()),
            Exp::Unary(op, exp) => {
                let factor = self.resolve_exp(context, exp.as_ref())?;
                Ok(Exp::Unary(op.clone(), Box::new(factor)))
            }
            Exp::BinaryOperation(op, left, right) => {
                let left = self.resolve_exp(context, left.as_ref())?;
                let right = self.resolve_exp(context, right.as_ref())?;
                Ok(Exp::BinaryOperation(*op, Box::new(left), Box::new(right)))
            }
            Exp::Conditional(cond, then, else_) => {
                let cond = self.resolve_exp(context, cond.as_ref())?;
                let then = self.resolve_exp(context, then.as_ref())?;
                let else_ = self.resolve_exp(context, else_.as_ref())?;

                Ok(Exp::Conditional(
                    Box::new(cond),
                    Box::new(then),
                    Box::new(else_),
                ))
            }
        }
    }
    // fn resolve_factor(&mut self, context: &mut Context, factor: &Factor) -> anyhow::Result<Factor> {
    //     match factor {
    //         Exp::Constant(_) => Ok(factor.clone()),
    //         Exp::Unary(op, factor) => {
    //             let factor = self.resolve_factor(context, factor.as_ref())?;
    //             Ok(Factor::Unary(op.clone(), Box::new(factor)))
    //         }
    //     }
    // }

    fn resolve_declaration(
        &mut self,
        context: &mut Context,
        declaration: &Declaration,
    ) -> anyhow::Result<BlockItem> {
        if self.variable_map.contains_key(&declaration.name) {
            anyhow::bail!("Variable {} already declared", declaration.name);
        }

        let unique_name = context.next_var(&declaration.name);
        self.variable_map
            .insert(declaration.name.clone(), unique_name.clone());

        let init = match &declaration.init {
            Some(exp) => Some(self.resolve_exp(context, exp)?),
            None => None,
        };

        Ok(BlockItem::Declaration(Declaration {
            name: unique_name,
            init,
        }))
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
    use crate::parser::Function;

    use super::*;

    #[test]
    fn test_analysis() {
        let program = Program {
            function_definition: Function {
                name: "main".to_string(),
                body: vec![BlockItem::Declaration(Declaration {
                    name: "x".to_string(),
                    init: None,
                })],
            },
        };

        let mut analysis = Analysis::new(program);
        let program = analysis.run().unwrap();

        assert_eq!(program.function_definition.body.len(), 1);
        assert_eq!(
            program.function_definition.body[0],
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
                body: vec![
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
        };

        let mut analysis = Analysis::new(program);
        let res = analysis.run();

        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "Variable x already declared");
    }
}
