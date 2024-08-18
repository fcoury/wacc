#![allow(unused)]
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
};

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
        writeln!(f, "\tpushq\t%rbp")?;
        writeln!(f, "\tmovq\t%rsp, %rbp")?;
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
        writeln!(f, "\tmovl\t{}, {}", self.exp, self.reg)
    }
}

#[derive(Debug, PartialEq)]
pub struct Ret;

impl Display for Ret {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "movq\t%rbp, %rsp")?;
        writeln!(f, "\tpop\t%rbp")?;
        writeln!(f, "\tret")
    }
}

#[derive(Debug, PartialEq)]
pub enum Reg {
    AX,
    R10,
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Reg::AX => write!(f, "%eax"),
            Reg::R10 => write!(f, "%r10d"),
        }
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
            Instruction::Mov(src, dst) => write!(f, "movl\t{}, {}", src, dst),
            Instruction::Unary(operator, operand) => write!(f, "{}\t{}", operator, operand),
            Instruction::AllocateStack(size) => write!(f, "subq\t${}, %rsp", size),
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

impl Display for UnaryOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            UnaryOperator::Neg => write!(f, "negl"),
            UnaryOperator::Not => write!(f, "notl"),
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
            Operand::Imm(imm) => write!(f, "${}", imm),
            Operand::Reg(reg) => write!(f, "{}", reg),
            Operand::Pseudo(pseudo) => write!(f, "{}", pseudo),
            Operand::Stack(offset) => write!(f, "{}(%rbp)", offset),
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

    pub fn run(&self) -> anyhow::Result<String> {
        let program = self.program.clone().into();
        let program = self.replace_pseudoregisters(program);
        let program = self.fix_instructions(program);
        Ok(program.to_string())
    }

    pub fn fix_instructions(&self, program: Program) -> Program {
        let instructions = program
            .function_definition
            .instructions
            .into_iter()
            .map(|instruction| match instruction {
                Instruction::Mov(src, dst) => match (src, dst) {
                    (Operand::Stack(src), Operand::Stack(dst)) => {
                        vec![
                            Instruction::Mov(Operand::Stack(src), Operand::Reg(Reg::R10)),
                            Instruction::Mov(Operand::Reg(Reg::R10), Operand::Stack(dst)),
                        ]
                    }
                    (src, dst) => vec![Instruction::Mov(src, dst)],
                },
                instr => vec![instr],
            })
            .flatten()
            .collect();

        Program {
            function_definition: Function {
                name: program.function_definition.name,
                instructions,
            },
        }
    }

    pub fn replace_pseudoregisters(&self, program: Program) -> Program {
        let mut stack_offset = 0;
        let mut stack_map = HashMap::new();

        let mut instructions = program
            .function_definition
            .instructions
            .into_iter()
            .map(|instruction| match instruction {
                Instruction::Mov(src, dst) => {
                    let src = match src {
                        Operand::Pseudo(pseudo) => {
                            if let Some(offset) = stack_map.get(&pseudo) {
                                Operand::Stack(*offset)
                            } else {
                                stack_offset -= 4;
                                stack_map.insert(pseudo.clone(), stack_offset);
                                Operand::Stack(stack_offset)
                            }
                        }
                        _ => src,
                    };

                    let dst = match dst {
                        Operand::Pseudo(pseudo) => {
                            if let Some(offset) = stack_map.get(&pseudo) {
                                Operand::Stack(*offset)
                            } else {
                                stack_offset -= 4;
                                stack_map.insert(pseudo.clone(), stack_offset);
                                Operand::Stack(stack_offset)
                            }
                        }
                        _ => dst,
                    };

                    Instruction::Mov(src, dst)
                }
                Instruction::Unary(op, operand) => {
                    let operand = match operand {
                        Operand::Pseudo(pseudo) => {
                            if let Some(offset) = stack_map.get(&pseudo) {
                                Operand::Stack(*offset)
                            } else {
                                stack_offset -= 4;
                                stack_map.insert(pseudo.clone(), stack_offset);
                                Operand::Stack(stack_offset)
                            }
                        }
                        _ => operand,
                    };

                    Instruction::Unary(op, operand)
                }
                Instruction::AllocateStack(size) => {
                    stack_offset -= size;
                    Instruction::AllocateStack(size)
                }
                Instruction::Ret(ret) => Instruction::Ret(ret),
            })
            .collect::<Vec<_>>();

        instructions.insert(0, Instruction::AllocateStack(-stack_offset));

        let new_program = Program {
            function_definition: Function {
                name: program.function_definition.name,
                instructions,
            },
        };

        new_program
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assembly() {
        let ir = ir::Program {
            function_definition: ir::Function {
                name: "main".to_string(),
                instructions: vec![
                    ir::Instruction::Unary(
                        ir::UnaryOperator::Negate,
                        ir::Val::Constant(8),
                        ir::Val::Var("tmp.0".to_string()),
                    ),
                    ir::Instruction::Unary(
                        ir::UnaryOperator::Complement,
                        ir::Val::Var("tmp.0".to_string()),
                        ir::Val::Var("tmp.1".to_string()),
                    ),
                    ir::Instruction::Unary(
                        ir::UnaryOperator::Negate,
                        ir::Val::Var("tmp.1".to_string()),
                        ir::Val::Var("tmp.2".to_string()),
                    ),
                    ir::Instruction::Return(ir::Val::Var("tmp.2".to_string())),
                ],
            },
        };

        let assembler = Assembler::new(ir);
        let program = assembler.run().unwrap();
        println!("{}", program);
        // assert_eq!(
        //     program.function_definition,
        //     Function {
        //         name: "main".to_string(),
        //         instructions: vec![
        //             Instruction::AllocateStack(12),
        //             Instruction::Mov(Operand::Imm(8), Operand::Stack(-4)),
        //             Instruction::Unary(UnaryOperator::Neg, Operand::Stack(-4)),
        //             Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
        //             Instruction::Unary(UnaryOperator::Not, Operand::Stack(-8)),
        //             Instruction::Mov(Operand::Stack(-8), Operand::Stack(-12)),
        //             Instruction::Unary(UnaryOperator::Neg, Operand::Stack(-12)),
        //             Instruction::Mov(Operand::Stack(-12), Operand::Reg(Reg::AX)),
        //             Instruction::Ret(Ret),
        //         ]
        //     }
        // );
    }

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
