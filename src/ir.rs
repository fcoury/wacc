use std::sync::atomic::{AtomicI32, Ordering};

use crate::parser::{self, BlockItem};

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
        let mut context = Context::new();
        Function {
            name: function.name.clone(),
            instructions: function.into_instructions(&mut context),
        }
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

impl IntoInstructions for parser::Function {
    fn into_instructions(self, context: &mut Context) -> Vec<Instruction> {
        let mut instructions = self
            .body
            .into_iter()
            .map(|item| match item {
                BlockItem::Statement(statement) => statement.clone().into_instructions(context),
                BlockItem::Declaration(declaration) => {
                    declaration.clone().into_instructions(context)
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        // if no return is found
        if let Some(last) = instructions.last() {
            if !matches!(last, Instruction::Return(_)) {
                instructions.push(Instruction::Return(Val::Constant(0)));
            }
        } else {
            instructions.push(Instruction::Return(Val::Constant(0)));
        }
        instructions
    }
}

impl IntoInstructions for parser::Exp {
    fn into_instructions(self, context: &mut Context) -> Vec<Instruction> {
        let mut instructions = vec![];
        emit_ir(self, &mut instructions, context);
        instructions
    }
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
            parser::Statement::Expression(exp) => {
                let mut instructions = vec![];
                emit_ir(exp, &mut instructions, context);
                instructions
            }
            parser::Statement::Null => vec![],
            parser::Statement::If(cond, then, else_) => {
                let mut instructions = vec![];
                let cond = emit_ir(cond, &mut instructions, context);
                let end_label = context.next_label("end");
                let else_label = context.next_label("else");

                instructions.push(Instruction::JumpIfZero(cond.clone(), else_label.clone()));
                let then_instructions = then.into_instructions(context);
                instructions.extend(then_instructions);
                instructions.push(Instruction::Jump(end_label.clone()));
                instructions.push(Instruction::Label(else_label.clone()));
                if let Some(else_) = else_ {
                    let else_instructions = else_.into_instructions(context);
                    instructions.extend(else_instructions);
                }
                instructions.push(Instruction::Label(end_label.clone()));
                instructions
            }
            parser::Statement::Compound(_) => todo!(),
        }
    }
}

impl IntoInstructions for parser::Declaration {
    fn into_instructions(self, context: &mut Context) -> Vec<Instruction> {
        let Some(init) = self.init else {
            return vec![];
        };

        let mut instructions = vec![];
        emit_assignment(self.name, init, &mut instructions, context);
        instructions
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Constant(i32),
    Var(Identifier),
}

pub fn emit_assignment(
    name: String,
    exp: parser::Exp,
    instructions: &mut Vec<Instruction>,
    context: &mut Context,
) -> Val {
    let val = emit_ir(exp, instructions, context);
    instructions.push(Instruction::Copy(val.clone(), Val::Var(name)));
    val
}

pub fn emit_ir(
    exp: parser::Exp,
    instructions: &mut Vec<Instruction>,
    context: &mut Context,
) -> Val {
    match exp {
        parser::Exp::Var(name) => Val::Var(name),
        parser::Exp::Assignment(exp, rhs) => match exp.as_ref() {
            parser::Exp::Var(name) => emit_assignment(name.clone(), *rhs, instructions, context),
            _ => todo!(),
        },
        parser::Exp::Constant(value) => Val::Constant(value),
        parser::Exp::Unary(operator, exp) => {
            let src = emit_ir(*exp, instructions, context);
            let dst = Val::Var(context.next_var());
            instructions.push(Instruction::Unary(operator.into(), src, dst.clone()));
            dst
        }
        parser::Exp::BinaryOperation(oper, v1, v2) => match oper {
            parser::BinaryOperator::Or => {
                let short_circuit_label = context.next_label("short_circuit");
                let end_label = context.next_label("end");
                let result = context.next_var();

                // Evaluate first operand
                let val1 = emit_ir(*v1, instructions, context);

                // if true (not 0), short-circuit
                instructions.push(Instruction::JumpIfNotZero(
                    val1.clone(),
                    short_circuit_label.clone(),
                ));

                // Evaluate second operand
                let val2 = emit_ir(*v2, instructions, context);

                // if true (not 0), short-circuit
                instructions.push(Instruction::JumpIfNotZero(
                    val2.clone(),
                    short_circuit_label.clone(),
                ));

                // Set result based on second operand
                instructions.push(Instruction::Copy(
                    Val::Constant(0),
                    Val::Var(result.clone()),
                ));
                // And jump to end
                instructions.push(Instruction::Jump(end_label.clone()));

                // False label
                instructions.push(Instruction::Label(short_circuit_label.clone()));
                instructions.push(Instruction::Copy(
                    Val::Constant(1),
                    Val::Var(result.clone()),
                ));

                // End label
                instructions.push(Instruction::Label(end_label.clone()));

                Val::Var(result)
            }
            parser::BinaryOperator::And => {
                let short_circuit_label = context.next_label("short_circuit");
                let end_label = context.next_label("end");
                let result = context.next_var();

                // Evaluate first operand
                let val1 = emit_ir(*v1, instructions, context);

                // For AND: if false (0), short-circuit
                instructions.push(Instruction::JumpIfZero(
                    val1.clone(),
                    short_circuit_label.clone(),
                ));

                // Evaluate second operand
                let val2 = emit_ir(*v2, instructions, context);

                // For AND: if false (0), short-circuit
                instructions.push(Instruction::JumpIfZero(
                    val2.clone(),
                    short_circuit_label.clone(),
                ));

                // Set result based on second operand
                instructions.push(Instruction::Copy(
                    Val::Constant(1),
                    Val::Var(result.clone()),
                ));
                // And jump to end
                instructions.push(Instruction::Jump(end_label.clone()));

                // False label
                instructions.push(Instruction::Label(short_circuit_label.clone()));
                instructions.push(Instruction::Copy(
                    Val::Constant(0),
                    Val::Var(result.clone()),
                ));

                // End label
                instructions.push(Instruction::Label(end_label.clone()));

                Val::Var(result)
            }
            _ => {
                let val1 = emit_ir(*v1, instructions, context);
                let val2 = emit_ir(*v2, instructions, context);
                let dst = Val::Var(context.next_var());
                instructions.push(Instruction::Binary(oper.into(), val1, val2, dst.clone()));
                dst
            }
        },
        parser::Exp::Conditional(cond, e1, e2) => {
            // Emit instructions for condition
            let c = emit_ir(*cond, instructions, context);

            // Create labels
            let e2_label = context.next_label("else");
            let end_label = context.next_label("end");

            // Create result variable
            let result = context.next_var();

            // Jump to e2 if condition is zero
            instructions.push(Instruction::JumpIfZero(c, e2_label.clone()));

            // Emit instructions for e1
            let v1 = emit_ir(*e1, instructions, context);

            // Assign result of e1 to result
            instructions.push(Instruction::Copy(v1, Val::Var(result.clone())));

            // Jump to end
            instructions.push(Instruction::Jump(end_label.clone()));

            // Label for e2
            instructions.push(Instruction::Label(e2_label));

            // Emit instructions for e2
            let v2 = emit_ir(*e2, instructions, context);

            // Assign result of e2 to result
            instructions.push(Instruction::Copy(v2, Val::Var(result.clone())));

            // End label
            instructions.push(Instruction::Label(end_label));

            // Return result
            Val::Var(result)
        }
    }
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

    pub fn next_label(&self, descr: &str) -> Identifier {
        let id = self.next_temp.fetch_add(1, Ordering::SeqCst);
        format!("{descr}.label.{}", id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{self, Block, BlockItem};

    #[test]
    fn test_constant() {
        let program = parser::Program {
            function_definition: parser::Function {
                name: "main".to_string(),
                body: Block {
                    items: vec![BlockItem::Statement(parser::Statement::Return(
                        parser::Exp::Constant(3),
                    ))],
                },
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
    fn test_unary_var_var() {
        let program = parser::Program {
            function_definition: parser::Function {
                name: "main".to_string(),
                body: Block {
                    items: vec![BlockItem::Statement(parser::Statement::Return(
                        parser::Exp::Unary(
                            parser::UnaryOperator::Negate,
                            Box::new(parser::Exp::Var("a".to_string())),
                        ),
                    ))],
                },
            },
        };

        let program = Ir::new(program).run();
        let instr = program.function_definition.instructions;
        println!("{:?}", instr);
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
