use std::sync::atomic::{AtomicI32, Ordering};

use crate::parser;

pub type Identifier = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub function_definition: Function,
}

impl From<parser::Program> for Program {
    fn from(program: parser::Program) -> Self {
        Program {
            function_definition: program.function_definition.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Identifier,
    pub instructions: Vec<Instruction>,
}

impl From<parser::Function> for Function {
    fn from(function: parser::Function) -> Self {
        let mut context = Context::new();
        Function {
            name: function.name,
            instructions: function.body.into_instructions(&mut context),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Return(Val),
    Unary(UnaryOperator, Val, Val),
}

trait IntoInstructions {
    fn into_instructions(self, context: &mut Context) -> Vec<Instruction>;
}

impl IntoInstructions for parser::Statement {
    fn into_instructions(self, context: &mut Context) -> Vec<Instruction> {
        match self {
            parser::Statement::Return(exp) => {
                let mut instructions = vec![];
                let val = emit_ir(exp, &mut instructions, context);
                instructions.push(Instruction::Return(val));
                instructions
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Constant(i32),
    Var(Identifier),
}

pub fn emit_ir(
    exp: parser::Exp,
    instructions: &mut Vec<Instruction>,
    context: &mut Context,
) -> Val {
    match exp {
        parser::Exp::Constant(value) => Val::Constant(value),
        parser::Exp::Unary(operator, exp) => {
            let src = emit_ir(*exp, instructions, context);
            let dst = Val::Var(context.next_var());
            instructions.push(Instruction::Unary(operator.into(), src, dst.clone()));
            dst
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Complement,
}

impl From<parser::UnaryOperator> for UnaryOperator {
    fn from(operator: parser::UnaryOperator) -> Self {
        match operator {
            parser::UnaryOperator::Negate => UnaryOperator::Negate,
            parser::UnaryOperator::Complement => UnaryOperator::Complement,
        }
    }
}

pub struct Ir {
    program: crate::parser::Program,
}

impl Ir {
    pub fn new(program: crate::parser::Program) -> Self {
        Self { program }
    }

    pub fn run(self) -> Program {
        self.program.into()
    }
}

// New struct to manage context
pub struct Context {
    next_temp: AtomicI32,
}

impl Context {
    pub fn new() -> Self {
        Self {
            next_temp: AtomicI32::new(0),
        }
    }

    pub fn next_var(&self) -> Identifier {
        let id = self.next_temp.fetch_add(1, Ordering::SeqCst);
        format!("tmp.{}", id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    #[test]
    fn test_constant() {
        let program = parser::Program {
            function_definition: parser::Function {
                name: "main".to_string(),
                body: parser::Statement::Return(parser::Exp::Constant(3)),
            },
        };
        let program = Ir::new(program).run();
        let instr = program.function_definition.instructions;

        assert_eq!(
            instr.get(0).unwrap(),
            &Instruction::Return(Val::Constant(3))
        );
    }

    #[test]
    fn test_unary() {
        let program = parser::Program {
            function_definition: parser::Function {
                name: "main".to_string(),
                body: parser::Statement::Return(parser::Exp::Unary(
                    parser::UnaryOperator::Complement,
                    Box::new(parser::Exp::Constant(2)),
                )),
            },
        };
        let program = Ir::new(program).run();
        let instr = program.function_definition.instructions;

        assert_eq!(
            instr,
            vec![
                Instruction::Unary(
                    UnaryOperator::Complement,
                    Val::Constant(2),
                    Val::Var("tmp.0".to_string())
                ),
                Instruction::Return(Val::Var("tmp.0".to_string()))
            ]
        );
    }

    #[test]
    fn test_multiple_unaries() {
        // -(~(-8))
        let program = parser::Program {
            function_definition: parser::Function {
                name: "main".to_string(),
                body: parser::Statement::Return(parser::Exp::Unary(
                    parser::UnaryOperator::Negate,
                    Box::new(parser::Exp::Unary(
                        parser::UnaryOperator::Complement,
                        Box::new(parser::Exp::Unary(
                            parser::UnaryOperator::Negate,
                            Box::new(parser::Exp::Constant(8)),
                        )),
                    )),
                )),
            },
        };
        let program = Ir::new(program).run();
        let instr = program.function_definition.instructions;

        assert_eq!(
            instr,
            vec![
                Instruction::Unary(
                    UnaryOperator::Negate,
                    Val::Constant(8),
                    Val::Var("tmp.0".to_string())
                ),
                Instruction::Unary(
                    UnaryOperator::Complement,
                    Val::Var("tmp.0".to_string()),
                    Val::Var("tmp.1".to_string())
                ),
                Instruction::Unary(
                    UnaryOperator::Negate,
                    Val::Var("tmp.1".to_string()),
                    Val::Var("tmp.2".to_string())
                ),
                Instruction::Return(Val::Var("tmp.2".to_string()))
            ]
        );
    }
}

