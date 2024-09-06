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

impl Program {
    pub fn iter(&self) -> std::slice::Iter<'_, Instruction> {
        self.function_definition.instructions.iter()
    }
}

impl From<ir::Program> for Program {
    fn from(program: ir::Program) -> Self {
        Program {
            function_definition: program.function_definition.into(),
        }
    }
}

impl Display for Program {
    #[cfg(not(target_os = "linux"))]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.function_definition)
    }

    #[cfg(target_os = "linux")]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.function_definition);
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
            .flat_map(Into::<Vec<Instruction>>::into)
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
            writeln!(f, "{}", instruction)?;
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
    CX,
    CL,
    DX,
    R10,
    R11,
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Reg::AX => write!(f, "%eax"),
            Reg::CX => write!(f, "%ecx"),
            Reg::CL => write!(f, "%cl"),
            Reg::DX => write!(f, "%edx"),
            Reg::R10 => write!(f, "%r10d"),
            Reg::R11 => write!(f, "%r11d"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Condition {
    E,
    NE,
    G,
    GE,
    L,
    LE,
}

impl Display for Condition {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Condition::E => write!(f, "e"),
            Condition::NE => write!(f, "ne"),
            Condition::G => write!(f, "g"),
            Condition::GE => write!(f, "ge"),
            Condition::L => write!(f, "l"),
            Condition::LE => write!(f, "le"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Mov(Operand, Operand),
    Unary(UnaryOperator, Operand),
    Binary(BinaryOperator, Operand, Operand),
    Cmp(Operand, Operand),
    Idiv(Operand),
    Cdq,
    Jmp(String),
    JmpCC(Condition, String),
    SetCC(Condition, Operand),
    Label(String),
    AllocateStack(i32),
    Ret(Ret),
}

impl From<ir::Instruction> for Vec<Instruction> {
    fn from(instruction: ir::Instruction) -> Self {
        match instruction {
            ir::Instruction::Unary(op, src, dst) => match op {
                ir::UnaryOperator::Not => vec![
                    Instruction::Cmp(Operand::Imm(0), src.into()),
                    Instruction::Mov(Operand::Imm(0), dst.clone().into()),
                    Instruction::SetCC(Condition::E, dst.into()),
                ],
                _ => vec![
                    Instruction::Mov(src.into(), dst.clone().into()),
                    Instruction::Unary(op.into(), dst.into()),
                ],
            },
            ir::Instruction::Return(val) => vec![
                Instruction::Mov(val.into(), Operand::Reg(Reg::AX)),
                Instruction::Ret(Ret),
            ],
            ir::Instruction::Binary(op, src1, src2, dst) => match op {
                ir::BinaryOperator::Add
                | ir::BinaryOperator::Subtract
                | ir::BinaryOperator::Multiply
                | ir::BinaryOperator::And
                | ir::BinaryOperator::Or
                | ir::BinaryOperator::BitwiseAnd
                | ir::BinaryOperator::BitwiseOr
                | ir::BinaryOperator::BitwiseXor => vec![
                    Instruction::Mov(src1.into(), dst.clone().into()),
                    Instruction::Binary(op.into(), src2.into(), dst.into()),
                ],
                ir::BinaryOperator::ShiftLeft | ir::BinaryOperator::ShiftRight => vec![
                    Instruction::Mov(src1.into(), dst.clone().into()),
                    Instruction::Mov(src2.into(), Operand::Reg(Reg::CX)),
                    Instruction::Binary(op.into(), Operand::Reg(Reg::CL), dst.into()),
                ],
                ir::BinaryOperator::Divide => vec![
                    Instruction::Mov(src1.into(), Operand::Reg(Reg::AX)),
                    Instruction::Cdq,
                    Instruction::Idiv(src2.into()),
                    Instruction::Mov(Operand::Reg(Reg::AX), dst.into()),
                ],
                ir::BinaryOperator::Remainder => vec![
                    Instruction::Mov(src1.into(), Operand::Reg(Reg::AX)),
                    Instruction::Cdq,
                    Instruction::Idiv(src2.into()),
                    Instruction::Mov(Operand::Reg(Reg::DX), dst.into()),
                ],
                ir::BinaryOperator::Equal => vec![
                    Instruction::Cmp(src1.into(), src2.into()),
                    Instruction::Mov(Operand::Imm(0), dst.clone().into()),
                    Instruction::SetCC(Condition::E, dst.into()),
                ],
                ir::BinaryOperator::NotEqual => vec![
                    Instruction::Cmp(src1.into(), src2.into()),
                    Instruction::Mov(Operand::Imm(0), dst.clone().into()),
                    Instruction::SetCC(Condition::NE, dst.into()),
                ],
                ir::BinaryOperator::GreaterThan => vec![
                    Instruction::Cmp(src2.into(), src1.into()),
                    Instruction::Mov(Operand::Imm(0), dst.clone().into()),
                    Instruction::SetCC(Condition::G, dst.into()),
                ],
                ir::BinaryOperator::GreaterOrEqual => vec![
                    Instruction::Cmp(src2.into(), src1.into()),
                    Instruction::Mov(Operand::Imm(0), dst.clone().into()),
                    Instruction::SetCC(Condition::GE, dst.into()),
                ],
                ir::BinaryOperator::LessThan => vec![
                    Instruction::Cmp(src2.into(), src1.into()),
                    Instruction::Mov(Operand::Imm(0), dst.clone().into()),
                    Instruction::SetCC(Condition::L, dst.into()),
                ],
                ir::BinaryOperator::LessOrEqual => vec![
                    Instruction::Cmp(src2.into(), src1.into()),
                    Instruction::Mov(Operand::Imm(0), dst.clone().into()),
                    Instruction::SetCC(Condition::LE, dst.into()),
                ],
                op => todo!("{:?}", op),
            },
            ir::Instruction::JumpIfZero(val, target) => vec![
                Instruction::Cmp(Operand::Imm(0), val.into()),
                Instruction::JmpCC(Condition::E, target),
            ],
            ir::Instruction::JumpIfNotZero(val, target) => vec![
                Instruction::Cmp(Operand::Imm(0), val.into()),
                Instruction::JmpCC(Condition::NE, target),
            ],
            ir::Instruction::Jump(target) => vec![Instruction::Jmp(target)],
            ir::Instruction::Label(label) => vec![Instruction::Label(label)],
            ir::Instruction::Copy(src, dst) => vec![Instruction::Mov(src.into(), dst.into())],
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Instruction::Mov(src, dst) => write!(f, "\tmovl\t{}, {}", src, dst),
            Instruction::Unary(operator, operand) => write!(f, "\t{}\t{}", operator, operand),
            Instruction::AllocateStack(size) => write!(f, "\tsubq\t${}, %rsp", size),
            Instruction::Ret(ret) => write!(f, "\t{}", ret),
            Instruction::Binary(operator, src1, src2) => {
                write!(f, "\t{}\t{}, {}", operator, src1, src2)
            }
            Instruction::Idiv(operand) => write!(f, "\tidivl\t{}", operand),
            Instruction::Cdq => write!(f, "\tcdq"),
            Instruction::Cmp(op1, op2) => write!(f, "\tcmpl\t{}, {}", op1, op2),
            Instruction::Jmp(target) => write!(f, "\tjmp\t.L{}", target),
            Instruction::JmpCC(condition, target) => write!(f, "\tj{}\t.L{}", condition, target),
            Instruction::SetCC(condition, operand) => {
                // TODO: implement this logic
                // if let Operand::Reg(reg) = operand {
                //     let reg = get_1byte_reg(reg);
                //     return write!(f, "\tset{}\t{}", condition, reg);
                // }
                write!(f, "\tset{}\t{}", condition, operand)
            }
            Instruction::Label(label) => write!(f, ".L{}:", label),
            _ => todo!(),
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
            ir::UnaryOperator::Not => todo!(),
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

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
}

impl From<ir::BinaryOperator> for BinaryOperator {
    fn from(operator: ir::BinaryOperator) -> Self {
        match operator {
            ir::BinaryOperator::Add => BinaryOperator::Add,
            ir::BinaryOperator::Subtract => BinaryOperator::Sub,
            ir::BinaryOperator::Multiply => BinaryOperator::Mul,
            ir::BinaryOperator::BitwiseAnd => BinaryOperator::And,
            ir::BinaryOperator::BitwiseOr => BinaryOperator::Or,
            ir::BinaryOperator::BitwiseXor => BinaryOperator::Xor,
            ir::BinaryOperator::ShiftLeft => BinaryOperator::ShiftLeft,
            ir::BinaryOperator::ShiftRight => BinaryOperator::ShiftRight,
            ir::BinaryOperator::And => BinaryOperator::And,
            ir::BinaryOperator::Or => BinaryOperator::Or,
            _ => todo!(),
        }
    }
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "addl"),
            BinaryOperator::Sub => write!(f, "subl"),
            BinaryOperator::Mul => write!(f, "imull"),
            BinaryOperator::And => write!(f, "andl"),
            BinaryOperator::Or => write!(f, "orl"),
            BinaryOperator::Xor => write!(f, "xorl"),
            BinaryOperator::ShiftLeft => write!(f, "sall"),
            BinaryOperator::ShiftRight => write!(f, "sarl"),
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

    pub fn assemble(&self) -> miette::Result<Program> {
        let program = self.program.clone().into();
        let program = self.replace_pseudoregisters(program);
        let program = self.fixup_instructions(program);
        Ok(program)
    }

    pub fn run(&self) -> miette::Result<String> {
        let program = self.assemble()?;
        Ok(program.to_string())
    }

    pub fn fixup_instructions(&self, program: Program) -> Program {
        let instructions = program
            .function_definition
            .instructions
            .into_iter()
            .flat_map(|instruction| match instruction {
                Instruction::Mov(src, dst) => match (src, dst) {
                    (Operand::Stack(src), Operand::Stack(dst)) => {
                        vec![
                            Instruction::Mov(Operand::Stack(src), Operand::Reg(Reg::R10)),
                            Instruction::Mov(Operand::Reg(Reg::R10), Operand::Stack(dst)),
                        ]
                    }
                    (src, dst) => vec![Instruction::Mov(src, dst)],
                },
                Instruction::Idiv(operand) => vec![
                    Instruction::Mov(operand, Operand::Reg(Reg::R10)),
                    Instruction::Idiv(Operand::Reg(Reg::R10)),
                ],
                Instruction::Binary(op, src1, src2) => match (&op, src1, src2) {
                    (BinaryOperator::Add, Operand::Stack(src1), Operand::Stack(src2))
                    | (BinaryOperator::Sub, Operand::Stack(src1), Operand::Stack(src2))
                    | (BinaryOperator::ShiftLeft, Operand::Stack(src1), Operand::Stack(src2))
                    | (BinaryOperator::ShiftRight, Operand::Stack(src1), Operand::Stack(src2)) => {
                        vec![
                            Instruction::Mov(Operand::Stack(src1), Operand::Reg(Reg::R10)),
                            Instruction::Binary(op, Operand::Reg(Reg::R10), Operand::Stack(src2)),
                        ]
                    }
                    (BinaryOperator::Mul, src1, Operand::Stack(src2))
                    | (BinaryOperator::And, src1, Operand::Stack(src2))
                    | (BinaryOperator::Or, src1, Operand::Stack(src2))
                    | (BinaryOperator::Xor, src1, Operand::Stack(src2)) => vec![
                        Instruction::Mov(Operand::Stack(src2), Operand::Reg(Reg::R11)),
                        Instruction::Binary(op, src1, Operand::Reg(Reg::R11)),
                        Instruction::Mov(Operand::Reg(Reg::R11), Operand::Stack(src2)),
                    ],
                    (op, src1, src2) => vec![Instruction::Binary(*op, src1, src2)],
                },
                Instruction::Cmp(Operand::Stack(op1), Operand::Stack(op2)) => vec![
                    Instruction::Mov(Operand::Stack(op1), Operand::Reg(Reg::R10)),
                    Instruction::Cmp(Operand::Reg(Reg::R10), Operand::Stack(op2)),
                ],
                Instruction::Cmp(op1, Operand::Imm(op2)) => vec![
                    Instruction::Mov(Operand::Imm(op2), Operand::Reg(Reg::R11)),
                    Instruction::Cmp(op1, Operand::Reg(Reg::R11)),
                ],
                instr => vec![instr],
            })
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
                Instruction::Idiv(operand) => {
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

                    Instruction::Idiv(operand)
                }
                Instruction::Cdq => Instruction::Cdq,
                Instruction::Binary(op, src1, src2) => {
                    let src1 = match src1 {
                        Operand::Pseudo(pseudo) => {
                            if let Some(offset) = stack_map.get(&pseudo) {
                                Operand::Stack(*offset)
                            } else {
                                stack_offset -= 4;
                                stack_map.insert(pseudo.clone(), stack_offset);
                                Operand::Stack(stack_offset)
                            }
                        }
                        _ => src1,
                    };

                    let src2 = match src2 {
                        Operand::Pseudo(pseudo) => {
                            if let Some(offset) = stack_map.get(&pseudo) {
                                Operand::Stack(*offset)
                            } else {
                                stack_offset -= 4;
                                stack_map.insert(pseudo.clone(), stack_offset);
                                Operand::Stack(stack_offset)
                            }
                        }
                        _ => src2,
                    };

                    Instruction::Binary(op, src1, src2)
                }
                Instruction::Cmp(op1, op2) => {
                    let op1 = match op1 {
                        Operand::Pseudo(pseudo) => {
                            if let Some(offset) = stack_map.get(&pseudo) {
                                Operand::Stack(*offset)
                            } else {
                                stack_offset -= 4;
                                stack_map.insert(pseudo.clone(), stack_offset);
                                Operand::Stack(stack_offset)
                            }
                        }
                        _ => op1,
                    };

                    let op2 = match op2 {
                        Operand::Pseudo(pseudo) => {
                            if let Some(offset) = stack_map.get(&pseudo) {
                                Operand::Stack(*offset)
                            } else {
                                stack_offset -= 4;
                                stack_map.insert(pseudo.clone(), stack_offset);
                                Operand::Stack(stack_offset)
                            }
                        }
                        _ => op2,
                    };

                    Instruction::Cmp(op1, op2)
                }
                Instruction::SetCC(condition, operand) => {
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

                    Instruction::SetCC(condition, operand)
                }
                Instruction::Ret(ret) => Instruction::Ret(ret),
                _ => instruction,
            })
            .collect::<Vec<_>>();

        instructions.insert(0, Instruction::AllocateStack(-stack_offset));

        Program {
            function_definition: Function {
                name: program.function_definition.name,
                instructions,
            },
        }
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

    fn ir_prog(instrs: Vec<ir::Instruction>) -> ir::Program {
        ir::Program {
            function_definition: ir::Function {
                name: "main".to_string(),
                instructions: instrs,
            },
        }
    }

    #[test]
    fn test_pseudo_and_vars() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::And,
            ir::Val::Var("tmp.0".to_string()),
            ir::Val::Var("tmp.1".to_string()),
            ir::Val::Var("tmp.2".to_string()),
        )]);

        let asm = Assembler::new(prog.clone());
        let prog = asm.replace_pseudoregisters(prog.into());

        let expected = vec![
            Instruction::AllocateStack(12),
            Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
            Instruction::Binary(BinaryOperator::And, Operand::Stack(-12), Operand::Stack(-8)),
        ];

        assert_eq!(prog.function_definition.instructions, expected);
    }

    #[test]
    fn test_fixup_and() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::And,
            ir::Val::Var("tmp.0".to_string()),
            ir::Val::Var("tmp.1".to_string()),
            ir::Val::Var("tmp.2".to_string()),
        )]);

        let asm = Assembler::new(prog.clone());
        let prog = asm.replace_pseudoregisters(prog.into());

        let expected = vec![
            Instruction::AllocateStack(12),
            Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
            Instruction::Binary(BinaryOperator::And, Operand::Stack(-12), Operand::Stack(-8)),
        ];

        assert_eq!(prog.function_definition.instructions, expected);
    }
    #[test]
    fn test_pseudo_mul_vars() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::Multiply,
            ir::Val::Var("tmp.0".to_string()),
            ir::Val::Var("tmp.1".to_string()),
            ir::Val::Var("tmp.2".to_string()),
        )]);

        let asm = Assembler::new(prog.clone());
        let prog = asm.replace_pseudoregisters(prog.into());

        let expected = vec![
            Instruction::AllocateStack(12),
            Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
            Instruction::Binary(BinaryOperator::Mul, Operand::Stack(-12), Operand::Stack(-8)),
        ];

        assert_eq!(prog.function_definition.instructions, expected);
    }

    #[test]
    fn test_pseudo_mul_const_var() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::Multiply,
            ir::Val::Constant(3),
            ir::Val::Var("tmp.1".to_string()),
            ir::Val::Var("tmp.2".to_string()),
        )]);

        let asm = Assembler::new(prog.clone());
        let prog = asm.replace_pseudoregisters(prog.into());

        let expected = vec![
            Instruction::AllocateStack(8),
            Instruction::Mov(Operand::Imm(3), Operand::Stack(-4)),
            Instruction::Binary(BinaryOperator::Mul, Operand::Stack(-8), Operand::Stack(-4)),
        ];

        assert_eq!(prog.function_definition.instructions, expected);

        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::Multiply,
            ir::Val::Var("tmp.1".to_string()),
            ir::Val::Constant(3),
            ir::Val::Var("tmp.2".to_string()),
        )]);

        let asm = Assembler::new(prog.clone());
        let prog = asm.replace_pseudoregisters(prog.into());

        let expected = vec![
            Instruction::AllocateStack(8),
            Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
            Instruction::Binary(BinaryOperator::Mul, Operand::Imm(3), Operand::Stack(-8)),
        ];

        assert_eq!(prog.function_definition.instructions, expected);
    }

    #[test]
    fn test_imull_gen() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::Multiply,
            ir::Val::Constant(3),
            ir::Val::Var("tmp.0".to_string()),
            ir::Val::Var("tmp.1".to_string()),
        )]);

        let assembler = Assembler::new(prog.clone());
        let program = assembler.run().unwrap();
        println!("{:#?}", program);
        println!("{}", program);
        // assert_eq!(
        //     program.function_definition,
        //     Function {
        //         name: "main".to_string(),
        //         instructions: vec![
        //             Instruction::Mov(Operand::Imm(42), Operand::Pseudo("tmp.0".to_string())),
        //             Instruction::Unary(UnaryOperator::Neg, Operand::Pseudo("tmp.0".to_string())),
        //         ]
        //     }
        // );
    }

    #[test]
    fn test_andl_gen() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::And,
            ir::Val::Constant(3),
            ir::Val::Var("tmp.0".to_string()),
            ir::Val::Var("tmp.1".to_string()),
        )]);

        let assembler = Assembler::new(prog.clone());
        let program = assembler.run().unwrap();
        println!("{:#?}", program);
        println!("{}", program);
    }

    #[test]
    fn test_orl_gen() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::Or,
            ir::Val::Constant(3),
            ir::Val::Var("tmp.0".to_string()),
            ir::Val::Var("tmp.1".to_string()),
        )]);

        let assembler = Assembler::new(prog.clone());
        let program = assembler.run().unwrap();
        println!("{:#?}", program);
        println!("{}", program);
    }

    #[test]
    fn test_xorl_gen() {
        let prog = ir_prog(vec![ir::Instruction::Binary(
            ir::BinaryOperator::BitwiseXor,
            ir::Val::Constant(3),
            ir::Val::Var("tmp.0".to_string()),
            ir::Val::Var("tmp.1".to_string()),
        )]);

        let assembler = Assembler::new(prog.clone());
        let program = assembler.run().unwrap();
        println!("{:#?}", program);
        println!("{}", program);
    }
}
