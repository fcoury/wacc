#![allow(unused)]
use std::fmt::{self, Display, Formatter};

use crate::ir;

#[derive(Debug, PartialEq)]
pub struct Program {
    function_definition: Function,
}

impl From<ir::Program> for Program {
    fn from(program: ir::Program) -> Self {
        Program {
            function_definition: program.function_definition.into(),
        }
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.function_definition);
        #[cfg(target_os = "linux")]
        writeln!(f, "\t.section .note.GNU-stack,\"\",@progbits")
    }
}

#[derive(Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
}

impl From<ir::Function> for Function {
    fn from(function: ir::Function) -> Self {
        let instructions: Vec<Instruction> = function
            .instructions
            .into_iter()
            .map(Into::<Vec<Instruction>>::into)
            .flatten()
            .collect();

        Function {
            name: function.name,
            instructions,
        }
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "\t.globl {}", self.name)?;
        writeln!(f, "{}:", self.name)?;
        for instruction in &self.instructions {
            writeln!(f, "\t{}", instruction)?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Mov {
    pub exp: Operand,
    pub reg: Reg,
}

impl Display for Mov {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "movl {}, {}", self.exp, self.reg)
    }
}

#[derive(Debug, PartialEq)]
pub struct Ret;

impl Display for Ret {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "ret")
    }
}

#[derive(Debug, PartialEq)]
pub enum Reg {
    AX,
    R10,
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Mov(Operand, Operand),
    Unary(UnaryOperator, Operand),
    AllocateStack(i32),
    Ret(Ret),
}

impl From<ir::Instruction> for Vec<Instruction> {
    fn from(instruction: ir::Instruction) -> Self {
        match instruction {
            ir::Instruction::Unary(op, src, dst) => vec![
                Instruction::Mov(src.into(), dst.clone().into()),
                Instruction::Unary(op.into(), dst.into()),
            ],
            ir::Instruction::Return(val) => vec![
                Instruction::Mov(val.into(), Operand::Reg(Reg::AX)),
                Instruction::Ret(Ret),
            ],
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Instruction::Mov(src, dst) => todo!(),
            Instruction::Unary(_, _) => todo!(),
            Instruction::AllocateStack(size) => todo!(),
            Instruction::Ret(ret) => write!(f, "{}", ret),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum UnaryOperator {
    Neg,
    Not,
}

impl From<ir::UnaryOperator> for UnaryOperator {
    fn from(operator: ir::UnaryOperator) -> Self {
        match operator {
            ir::UnaryOperator::Negate => UnaryOperator::Neg,
            ir::UnaryOperator::Complement => UnaryOperator::Not,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Operand {
    Imm(i32),
    Reg(Reg),
    Pseudo(String),
    Stack(i32),
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Operand::Imm(imm) => write!(f, "{}", imm),
            Operand::Reg(reg) => todo!(),
            Operand::Pseudo(pseudo) => write!(f, "{}", pseudo),
            Operand::Stack(offset) => todo!(),
        }
    }
}

impl From<ir::Val> for Operand {
    fn from(val: ir::Val) -> Self {
        match val {
            ir::Val::Constant(value) => Operand::Imm(value),
            ir::Val::Var(identifier) => Operand::Pseudo(identifier),
        }
    }
}

pub struct Assembler {
    program: ir::Program,
}

impl Assembler {
    pub fn new(program: ir::Program) -> Assembler {
        Assembler { program }
    }

    pub fn run(self) -> anyhow::Result<Program> {
        Ok(self.program.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ret() {
        let ir = ir::Program {
            function_definition: ir::Function {
                name: "main".to_string(),
                instructions: vec![ir::Instruction::Return(ir::Val::Constant(42))],
            },
        };

        let program = Program::from(ir);
        assert_eq!(
            program.function_definition,
            Function {
                name: "main".to_string(),
                instructions: vec![
                    Instruction::Mov(Operand::Imm(42), Operand::Reg(Reg::AX)),
                    Instruction::Ret(Ret)
                ]
            }
        );
    }

    #[test]
    fn test_from_unary() {
        let ir = ir::Program {
            function_definition: ir::Function {
                name: "main".to_string(),
                instructions: vec![ir::Instruction::Unary(
                    ir::UnaryOperator::Negate,
                    ir::Val::Constant(42),
                    ir::Val::Var("tmp.0".to_string()),
                )],
            },
        };

        let program = Program::from(ir);
        assert_eq!(
            program.function_definition,
            Function {
                name: "main".to_string(),
                instructions: vec![
                    Instruction::Mov(Operand::Imm(42), Operand::Pseudo("tmp.0".to_string())),
                    Instruction::Unary(UnaryOperator::Neg, Operand::Pseudo("tmp.0".to_string())),
                ]
            }
        );
    }
}
