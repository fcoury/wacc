use std::sync::atomic::{AtomicI32, Ordering};

use crate::parser;

pub type Identifier = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub function_definition: Function,
}

impl Program {
    pub fn iter(&self) -> std::slice::Iter<'_, Instruction> {
        self.function_definition.instructions.iter()
    }
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
        todo!()
        // let mut context = Context::new();
        // Function {
        //     name: function.name,
        //     instructions: function.body.into_instructions(&mut context),
        // }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Return(Val),
    Unary(UnaryOperator, Val, Val),
    Binary(BinaryOperator, Val, Val, Val),
    Copy(Val, Val),
    Jump(Identifier),
    JumpIfZero(Val, Identifier),
    JumpIfNotZero(Val, Identifier),
    Label(Identifier),
}

trait IntoInstructions {
    fn into_instructions(self, context: &mut Context) -> Vec<Instruction>;
}

impl IntoInstructions for parser::Statement {
    fn into_instructions(self, context: &mut Context) -> Vec<Instruction> {
        todo!()
        // match self {
        //     parser::Statement::Return(exp) => {
        //         let mut instructions = vec![];
        //         let val = emit_ir(exp, &mut instructions, context);
        //         instructions.push(Instruction::Return(val));
        //         instructions
        //     }
        // }
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
    todo!()
    // match exp {
    //     parser::Exp::Factor(factor) => match factor {
    //         parser::Factor::Constant(value) => Val::Constant(value),
    //         parser::Factor::Unary(operator, exp) => {
    //             let src = emit_ir(parser::Exp::Factor(*exp), instructions, context);
    //             let dst = Val::Var(context.next_var());
    //             instructions.push(Instruction::Unary(operator.into(), src, dst.clone()));
    //             dst
    //         }
    //         parser::Factor::Exp(exp) => emit_ir(*exp, instructions, context),
    //     },
    //     parser::Exp::BinaryOperation(oper, v1, v2) => match oper {
    //         parser::BinaryOperator::Or => {
    //             let short_circuit_label = context.next_var();
    //             let end_label = context.next_var();
    //             let result = context.next_var();
    //
    //             // Evaluate first operand
    //             let val1 = emit_ir(*v1, instructions, context);
    //
    //             // if true (not 0), short-circuit
    //             instructions.push(Instruction::JumpIfNotZero(
    //                 val1.clone(),
    //                 short_circuit_label.clone(),
    //             ));
    //
    //             // Evaluate second operand
    //             let val2 = emit_ir(*v2, instructions, context);
    //
    //             // if true (not 0), short-circuit
    //             instructions.push(Instruction::JumpIfNotZero(
    //                 val2.clone(),
    //                 short_circuit_label.clone(),
    //             ));
    //
    //             // Set result based on second operand
    //             instructions.push(Instruction::Copy(
    //                 Val::Constant(0),
    //                 Val::Var(result.clone()),
    //             ));
    //             // And jump to end
    //             instructions.push(Instruction::Jump(end_label.clone()));
    //
    //             // False label
    //             instructions.push(Instruction::Label(short_circuit_label.clone()));
    //             instructions.push(Instruction::Copy(
    //                 Val::Constant(1),
    //                 Val::Var(result.clone()),
    //             ));
    //
    //             // End label
    //             instructions.push(Instruction::Label(end_label.clone()));
    //
    //             Val::Var(result)
    //         }
    //         parser::BinaryOperator::And => {
    //             let short_circuit_label = context.next_var();
    //             let end_label = context.next_var();
    //             let result = context.next_var();
    //
    //             // Evaluate first operand
    //             let val1 = emit_ir(*v1, instructions, context);
    //
    //             // For AND: if false (0), short-circuit
    //             instructions.push(Instruction::JumpIfZero(
    //                 val1.clone(),
    //                 short_circuit_label.clone(),
    //             ));
    //
    //             // Evaluate second operand
    //             let val2 = emit_ir(*v2, instructions, context);
    //
    //             // For AND: if false (0), short-circuit
    //             instructions.push(Instruction::JumpIfZero(
    //                 val2.clone(),
    //                 short_circuit_label.clone(),
    //             ));
    //
    //             // Set result based on second operand
    //             instructions.push(Instruction::Copy(
    //                 Val::Constant(1),
    //                 Val::Var(result.clone()),
    //             ));
    //             // And jump to end
    //             instructions.push(Instruction::Jump(end_label.clone()));
    //
    //             // False label
    //             instructions.push(Instruction::Label(short_circuit_label.clone()));
    //             instructions.push(Instruction::Copy(
    //                 Val::Constant(0),
    //                 Val::Var(result.clone()),
    //             ));
    //
    //             // End label
    //             instructions.push(Instruction::Label(end_label.clone()));
    //
    //             Val::Var(result)
    //         }
    //         _ => {
    //             let val1 = emit_ir(*v1, instructions, context);
    //             let val2 = emit_ir(*v2, instructions, context);
    //             let dst = Val::Var(context.next_var());
    //             instructions.push(Instruction::Binary(oper.into(), val1, val2, dst.clone()));
    //             dst
    //         }
    //     },
    // }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Complement,
    Not,
}

impl From<parser::UnaryOperator> for UnaryOperator {
    fn from(operator: parser::UnaryOperator) -> Self {
        match operator {
            parser::UnaryOperator::Negate => UnaryOperator::Negate,
            parser::UnaryOperator::Complement => UnaryOperator::Complement,
            parser::UnaryOperator::Not => UnaryOperator::Not,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}

impl From<parser::BinaryOperator> for BinaryOperator {
    fn from(operator: parser::BinaryOperator) -> Self {
        match operator {
            parser::BinaryOperator::Add => BinaryOperator::Add,
            parser::BinaryOperator::Subtract => BinaryOperator::Subtract,
            parser::BinaryOperator::Multiply => BinaryOperator::Multiply,
            parser::BinaryOperator::Divide => BinaryOperator::Divide,
            parser::BinaryOperator::Remainder => BinaryOperator::Remainder,
            parser::BinaryOperator::BitwiseAnd => BinaryOperator::BitwiseAnd,
            parser::BinaryOperator::BitwiseOr => BinaryOperator::BitwiseOr,
            parser::BinaryOperator::BitwiseXor => BinaryOperator::BitwiseXor,
            parser::BinaryOperator::ShiftLeft => BinaryOperator::ShiftLeft,
            parser::BinaryOperator::ShiftRight => BinaryOperator::ShiftRight,
            parser::BinaryOperator::And => BinaryOperator::And,
            parser::BinaryOperator::Or => BinaryOperator::Or,
            parser::BinaryOperator::Equal => BinaryOperator::Equal,
            parser::BinaryOperator::NotEqual => BinaryOperator::NotEqual,
            parser::BinaryOperator::LessThan => BinaryOperator::LessThan,
            parser::BinaryOperator::LessOrEqual => BinaryOperator::LessOrEqual,
            parser::BinaryOperator::GraterThan => BinaryOperator::GreaterThan,
            parser::BinaryOperator::GreaterOrEqual => BinaryOperator::GreaterOrEqual,
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
        // let program = parser::Program {
        //     function_definition: parser::Function {
        //         name: "main".to_string(),
        //         body: parser::Statement::Return(parser::Exp::Factor(parser::Factor::Constant(3))),
        //     },
        // };
        // let program = Ir::new(program).run();
        // let instr = program.function_definition.instructions;
        //
        // assert_eq!(
        //     instr.get(0).unwrap(),
        //     &Instruction::Return(Val::Constant(3))
        // );
    }

    #[test]
    fn test_unary() {
        // let program = parser::Program {
        //     function_definition: parser::Function {
        //         name: "main".to_string(),
        //         body: parser::Statement::Return(parser::Exp::Factor(parser::Factor::Unary(
        //             parser::UnaryOperator::Complement,
        //             Box::new(parser::Factor::Constant(2)),
        //         ))),
        //     },
        // };
        // let program = Ir::new(program).run();
        // let instr = program.function_definition.instructions;
        //
        // assert_eq!(
        //     instr,
        //     vec![
        //         Instruction::Unary(
        //             UnaryOperator::Complement,
        //             Val::Constant(2),
        //             Val::Var("tmp.0".to_string())
        //         ),
        //         Instruction::Return(Val::Var("tmp.0".to_string()))
        //     ]
        // );
    }

    #[test]
    fn test_multiple_unaries() {
        // // -(~(-8))
        // let program = parser::Program {
        //     function_definition: parser::Function {
        //         name: "main".to_string(),
        //         body: parser::Statement::Return(parser::Exp::Factor(parser::Factor::Unary(
        //             parser::UnaryOperator::Negate,
        //             Box::new(parser::Factor::Unary(
        //                 parser::UnaryOperator::Complement,
        //                 Box::new(parser::Factor::Unary(
        //                     parser::UnaryOperator::Negate,
        //                     Box::new(parser::Factor::Constant(8)),
        //                 )),
        //             )),
        //         ))),
        //     },
        // };
        // let program = Ir::new(program).run();
        // let instr = program.function_definition.instructions;
        //
        // assert_eq!(
        //     instr,
        //     vec![
        //         Instruction::Unary(
        //             UnaryOperator::Negate,
        //             Val::Constant(8),
        //             Val::Var("tmp.0".to_string())
        //         ),
        //         Instruction::Unary(
        //             UnaryOperator::Complement,
        //             Val::Var("tmp.0".to_string()),
        //             Val::Var("tmp.1".to_string())
        //         ),
        //         Instruction::Unary(
        //             UnaryOperator::Negate,
        //             Val::Var("tmp.1".to_string()),
        //             Val::Var("tmp.2".to_string())
        //         ),
        //         Instruction::Return(Val::Var("tmp.2".to_string()))
        //     ]
        // );
    }

    #[test]
    fn test_short_circuit() {
        // let program = parser::Program {
        //     function_definition: parser::Function {
        //         name: "main".to_string(),
        //         body: parser::Statement::Return(parser::Exp::BinaryOperation(
        //             parser::BinaryOperator::And,
        //             Box::new(parser::Exp::Factor(parser::Factor::Constant(1))),
        //             Box::new(parser::Exp::Factor(parser::Factor::Constant(2))),
        //         )),
        //     },
        // };
        // let program = Ir::new(program).run();
        // let instr = program.function_definition.instructions;
        //
        // assert_eq!(
        //     instr,
        //     vec![
        //         Instruction::JumpIfZero(Val::Constant(1), "tmp.0".to_string()),
        //         Instruction::JumpIfZero(Val::Constant(2), "tmp.0".to_string()),
        //         Instruction::Copy(Val::Constant(1), Val::Var("tmp.2".to_string())),
        //         Instruction::Jump("tmp.1".to_string()),
        //         Instruction::Label("tmp.0".to_string()),
        //         Instruction::Copy(Val::Constant(0), Val::Var("tmp.2".to_string())),
        //         Instruction::Label("tmp.1".to_string()),
        //         Instruction::Return(Val::Var("tmp.2".to_string()))
        //     ]
        // );
    }
}
