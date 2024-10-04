#![allow(unused)]
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
};

use crate::{
    common::ToCode,
    ir,
    lexer::Span,
    parser::Identifier,
    semantic::{InitialValue, SymbolMap, TypeInfo, VarAttrs},
    utils::safe_split_at,
};

#[derive(Debug, PartialEq)]
pub enum TopLevel {
    Function(Function),
    StaticVariable(StaticVar),
}

impl From<ir::TopLevel> for TopLevel {
    fn from(top_level: ir::TopLevel) -> Self {
        match top_level {
            ir::TopLevel::Function(function) => TopLevel::Function(function.into()),
            ir::TopLevel::StaticVariable(static_var) => TopLevel::StaticVariable(static_var.into()),
        }
    }
}

impl ToCode for TopLevel {
    fn to_code(&self, source: impl ToString, symbols: SymbolMap, f: &mut Formatter) -> fmt::Result {
        match self {
            TopLevel::Function(function) => function.to_code(source, symbols, f),
            TopLevel::StaticVariable(static_var) => write!(f, "{}", static_var),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct StaticVar {
    pub name: Identifier,
    pub global: bool,
    pub init: Operand,
}

impl From<ir::StaticVar> for StaticVar {
    fn from(value: ir::StaticVar) -> Self {
        StaticVar {
            name: value.name,
            global: value.global,
            init: value.init.into(),
        }
    }
}

impl fmt::Display for StaticVar {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

#[derive(Debug)]
pub struct Program {
    top_level: Vec<TopLevel>,
    symbols: SymbolMap,
}

impl Program {
    pub fn iter(&self) -> std::slice::Iter<'_, TopLevel> {
        self.top_level.iter()
    }
}

impl ToCode for Program {
    fn to_code(&self, source: impl ToString, symbols: SymbolMap, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "{}", self.symbols)?;

        for def in &self.top_level {
            def.to_code(source.to_string(), self.symbols.clone(), f)?;
        }

        if cfg!(target_os = "linux") {
            writeln!(f, "\t.section .note.GNU-stack,\"\",@progbits")?;
        }

        Ok(())
    }
}

impl Display for SymbolMap {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (name, scope) in self.iter() {
            if let TypeInfo::Variable(var) = &scope.info {
                if let VarAttrs::Static { init, global } = &var.attrs {
                    match init {
                        InitialValue::Initial(val) => {
                            if *global {
                                writeln!(f, "\t.global {}", name)?;
                            }
                            writeln!(f, "\t.data")?;
                            if cfg!(target_os = "macos") {
                                writeln!(f, "_{}:", name)?;
                            } else {
                                writeln!(f, "{}:", name)?;
                            }
                            writeln!(f, "\t.long {}", val)?;
                        }
                        _ => {
                            if *global {
                                writeln!(f, "\t.global {}", name)?;
                            }
                            writeln!(f, "\t.bss")?;
                            if cfg!(target_os = "macos") {
                                writeln!(f, "\t.balign 4")?;
                                writeln!(f, "_{}:", name)?;
                            } else {
                                writeln!(f, "\t.align 4")?;
                                writeln!(f, "{}:", name)?;
                            }
                            writeln!(f, "\t.zero 4")?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Function {
    pub name: Identifier,
    pub global: bool,
    pub instructions: Vec<Instruction>,
    pub stack_size: i32,
}

impl From<ir::Function> for Function {
    fn from(function: ir::Function) -> Self {
        let mut instructions = Vec::new();

        if function.instructions.is_empty() {
            return Function {
                name: function.name,
                global: function.global,
                instructions,
                stack_size: 0,
            };
        }

        for (i, param) in function.params.iter().enumerate() {
            if i < 6 {
                instructions.push(Instruction::Mov(
                    Operand::Reg(ARG_REGISTERS[i]),
                    Operand::Pseudo(param.clone()),
                ));
            } else {
                // we skip the first one because it's the return address, hence the -4 and not -5
                instructions.push(Instruction::Mov(
                    Operand::Stack((i as i32 - 4) * 8),
                    Operand::Pseudo(param.clone()),
                ));
            }
        }

        instructions.extend(
            function
                .instructions
                .iter()
                .flat_map(|instruction| from(instruction.clone(), &function))
                .collect::<Vec<_>>(),
        );

        Function {
            name: function.name,
            global: function.global,
            instructions,
            stack_size: function.params.len() as i32 * 8,
        }
    }
}

impl ToCode for Function {
    fn to_code(&self, source: impl ToString, symbols: SymbolMap, f: &mut Formatter) -> fmt::Result {
        if let Some(info) = symbols.get(&self.name) {
            // TODO: evaluate need to error check here
            if let Some(info) = info.as_function() {
                if info.attrs.global {
                    writeln!(f, "\t.global {}", self.name)?;
                }
            }
        }

        writeln!(f, "\t.text")?;
        if cfg!(target_os = "macos") {
            writeln!(f, "_{}:", self.name)?;
        } else {
            writeln!(f, "{}:", self.name)?;
        }
        writeln!(f, "\tpushq\t%rbp")?;
        writeln!(f, "\tmovq\t%rsp, %rbp")?;
        for instruction in &self.instructions {
            instruction.to_code(source.to_string(), symbols.clone(), f)?;
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

// Enum for the original registers
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Reg {
    AX,
    CX,
    DX,
    DI,
    SI,
    R8,
    R9,
    R10,
    R11,
}

impl From<Reg> for Reg8 {
    fn from(reg: Reg) -> Self {
        reg.as_8bit()
    }
}

impl From<Reg> for Reg16 {
    fn from(reg: Reg) -> Self {
        reg.as_16bit()
    }
}

impl From<Reg> for Reg32 {
    fn from(reg: Reg) -> Self {
        reg.as_32bit()
    }
}

impl From<Reg> for Reg64 {
    fn from(reg: Reg) -> Self {
        reg.as_64bit()
    }
}

impl Reg {
    fn as_8bit(&self) -> Reg8 {
        match self {
            Reg::AX => Reg8::AL,
            Reg::CX => Reg8::CL,
            Reg::DX => Reg8::DL,
            Reg::DI => Reg8::DIL,
            Reg::SI => Reg8::SIL,
            Reg::R8 => Reg8::R8B,
            Reg::R9 => Reg8::R9B,
            Reg::R10 => Reg8::R10B,
            Reg::R11 => Reg8::R11B,
        }
    }

    fn as_16bit(&self) -> Reg16 {
        match self {
            Reg::AX => Reg16::AX,
            Reg::CX => Reg16::CX,
            Reg::DX => Reg16::DX,
            Reg::DI => Reg16::DI,
            Reg::SI => Reg16::SI,
            Reg::R8 => Reg16::R8W,
            Reg::R9 => Reg16::R9W,
            Reg::R10 => Reg16::R10W,
            Reg::R11 => Reg16::R11W,
        }
    }

    fn as_32bit(&self) -> Reg32 {
        match self {
            Reg::AX => Reg32::EAX,
            Reg::CX => Reg32::ECX,
            Reg::DX => Reg32::EDX,
            Reg::DI => Reg32::EDI,
            Reg::SI => Reg32::ESI,
            Reg::R8 => Reg32::R8D,
            Reg::R9 => Reg32::R9D,
            Reg::R10 => Reg32::R10D,
            Reg::R11 => Reg32::R11D,
        }
    }

    fn as_64bit(&self) -> Reg64 {
        match self {
            Reg::AX => Reg64::RAX,
            Reg::CX => Reg64::RCX,
            Reg::DX => Reg64::RDX,
            Reg::DI => Reg64::RDI,
            Reg::SI => Reg64::RSI,
            Reg::R8 => Reg64::R8,
            Reg::R9 => Reg64::R9,
            Reg::R10 => Reg64::R10,
            Reg::R11 => Reg64::R11,
        }
    }
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Reg::AX => write!(f, "%eax"),
            Reg::CX => write!(f, "%ecx"),
            Reg::DX => write!(f, "%edx"),
            Reg::DI => write!(f, "%edi"),
            Reg::SI => write!(f, "%esi"),
            Reg::R8 => write!(f, "%r8d"),
            Reg::R9 => write!(f, "%r9d"),
            Reg::R10 => write!(f, "%r10d"),
            Reg::R11 => write!(f, "%r11d"),
        }
    }
}

// Enum for 8-bit registers
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Reg8 {
    AL,
    CL,
    DL,
    DIL,
    SIL,
    R8B,
    R9B,
    R10B,
    R11B,
}

impl Display for Reg8 {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Reg8::AL => write!(f, "%al"),
            Reg8::CL => write!(f, "%cl"),
            Reg8::DL => write!(f, "%dl"),
            Reg8::DIL => write!(f, "%dil"),
            Reg8::SIL => write!(f, "%sil"),
            Reg8::R8B => write!(f, "%r8b"),
            Reg8::R9B => write!(f, "%r9b"),
            Reg8::R10B => write!(f, "%r10b"),
            Reg8::R11B => write!(f, "%r11b"),
        }
    }
}

// Enum for 16-bit registers
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Reg16 {
    AX,
    CX,
    DX,
    DI,
    SI,
    R8W,
    R9W,
    R10W,
    R11W,
}

impl Display for Reg16 {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Reg16::AX => write!(f, "%ax"),
            Reg16::CX => write!(f, "%cx"),
            Reg16::DX => write!(f, "%dx"),
            Reg16::DI => write!(f, "%di"),
            Reg16::SI => write!(f, "%si"),
            Reg16::R8W => write!(f, "%r8w"),
            Reg16::R9W => write!(f, "%r9w"),
            Reg16::R10W => write!(f, "%r10w"),
            Reg16::R11W => write!(f, "%r11w"),
        }
    }
}

// Enum for 32-bit registers
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Reg32 {
    EAX,
    ECX,
    EDX,
    EDI,
    ESI,
    R8D,
    R9D,
    R10D,
    R11D,
}

impl Display for Reg32 {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Reg32::EAX => write!(f, "%eax"),
            Reg32::ECX => write!(f, "%ecx"),
            Reg32::EDX => write!(f, "%edx"),
            Reg32::EDI => write!(f, "%edi"),
            Reg32::ESI => write!(f, "%esi"),
            Reg32::R8D => write!(f, "%r8d"),
            Reg32::R9D => write!(f, "%r9d"),
            Reg32::R10D => write!(f, "%r10d"),
            Reg32::R11D => write!(f, "%r11d"),
        }
    }
}

// Enum for 64-bit registers
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Reg64 {
    RAX,
    RCX,
    RDX,
    RDI,
    RSI,
    R8,
    R9,
    R10,
    R11,
}

impl Display for Reg64 {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Reg64::RAX => write!(f, "%rax"),
            Reg64::RCX => write!(f, "%rcx"),
            Reg64::RDX => write!(f, "%rdx"),
            Reg64::RDI => write!(f, "%rdi"),
            Reg64::RSI => write!(f, "%rsi"),
            Reg64::R8 => write!(f, "%r8"),
            Reg64::R9 => write!(f, "%r9"),
            Reg64::R10 => write!(f, "%r10"),
            Reg64::R11 => write!(f, "%r11"),
        }
    }
}

const ARG_REGISTERS: [Reg; 6] = [Reg::DI, Reg::SI, Reg::DX, Reg::CX, Reg::R8, Reg::R9];

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
    Jmp(Identifier),
    JmpCC(Condition, Identifier),
    SetCC(Condition, Operand),
    Label(Identifier),
    AllocateStack(i32),
    DeallocateStack(i32),
    Push(Operand),
    Pop(Operand),
    Call(Identifier),
    Ret(Ret),
    SpanOnly(Span),
    // Comment(String),
}

// macro_rules! comment {
//     ($instructions:expr, $($arg:tt)*) => {
//         $instructions.push(Instruction::Comment(format!($($arg)*)));
//     };
// }

fn from(instruction: ir::Instruction, context: &ir::Function) -> Vec<Instruction> {
    match instruction {
        ir::Instruction::SpanOnly(span) => vec![Instruction::SpanOnly(span)],
        ir::Instruction::Unary(op, src, dst, span) => match op {
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
        ir::Instruction::Return(val, _span) => vec![
            Instruction::Mov(val.into(), Operand::Reg(Reg::AX)),
            Instruction::Ret(Ret),
        ],
        ir::Instruction::Binary(op, src1, src2, dst, span) => match op {
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
                Instruction::Binary(op.into(), Operand::Reg8(Reg8::CL), dst.into()),
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
        ir::Instruction::JumpIfZero(val, target, span) => vec![
            Instruction::Cmp(Operand::Imm(0), val.into()),
            Instruction::JmpCC(Condition::E, target),
        ],
        ir::Instruction::JumpIfNotZero(val, target, span) => vec![
            Instruction::Cmp(Operand::Imm(0), val.into()),
            Instruction::JmpCC(Condition::NE, target),
        ],
        ir::Instruction::Jump(target, span) => {
            vec![Instruction::Jmp(target)]
        }
        ir::Instruction::Label(label, span) => {
            vec![Instruction::Label(label)]
        }
        ir::Instruction::Copy(src, dst, span) => vec![Instruction::Mov(src.into(), dst.into())],
        ir::Instruction::FunCall(fun_name, args, dst, span) => {
            let mut instructions = vec![];

            // adjust stack alignment
            let (register_args, stack_args) = safe_split_at(&args, 6);
            let stack_padding = if stack_args.len() % 2 == 0 { 0 } else { 8 };

            if stack_padding > 0 {
                instructions.push(Instruction::AllocateStack(stack_padding));
            }

            // pass args in registers
            for (reg_index, tacky_arg) in register_args.iter().enumerate() {
                let r = ARG_REGISTERS[reg_index];
                instructions.push(Instruction::Mov(tacky_arg.clone().into(), Operand::Reg(r)));
            }

            // pass args on stack
            for tacky_arg in stack_args.iter().rev() {
                let assembly_arg = tacky_arg.clone().into();
                match assembly_arg {
                    Operand::Imm(_) => instructions.push(Instruction::Push(assembly_arg)),
                    Operand::Reg(_) => instructions.push(Instruction::Push(assembly_arg)),
                    _ => {
                        instructions.push(Instruction::Mov(assembly_arg, Operand::Reg(Reg::AX)));
                        instructions.push(Instruction::Push(Operand::Reg(Reg::AX)));
                    }
                }
            }

            // emit call instruction
            instructions.push(Instruction::Call(fun_name));

            // adjust stack pointer
            let bytes_to_remove = 8 * stack_args.len() as i32 + stack_padding;
            if bytes_to_remove > 0 {
                instructions.push(Instruction::DeallocateStack(bytes_to_remove));
            }

            // retrieve return value
            instructions.push(Instruction::Mov(Operand::Reg(Reg::AX), dst.into()));
            instructions
        }
    }
}

impl ToCode for Instruction {
    fn to_code(
        &self,
        source: impl ToString,
        sumbols: SymbolMap,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let source = source.to_string();
        match self {
            Instruction::Mov(src, dst) => writeln!(f, "\tmovl\t{}, {}", src, dst),
            Instruction::Unary(operator, operand) => writeln!(f, "\t{}\t{}", operator, operand),
            Instruction::AllocateStack(size) => writeln!(f, "\tsubq\t${}, %rsp", size),
            Instruction::Ret(ret) => writeln!(f, "\t{}", ret),
            Instruction::Binary(operator, src1, src2) => {
                writeln!(f, "\t{}\t{}, {}", operator, src1, src2)
            }
            Instruction::Idiv(operand) => writeln!(f, "\tidivl\t{}", operand),
            Instruction::Cdq => writeln!(f, "\tcdq"),
            Instruction::Cmp(op1, op2) => writeln!(f, "\tcmpl\t{}, {}", op1, op2),
            Instruction::Jmp(target) => writeln!(f, "\tjmp\t.L{}", target),
            Instruction::JmpCC(condition, target) => writeln!(f, "\tj{}\t.L{}", condition, target),
            Instruction::SetCC(condition, operand) => {
                // TODO: implement this logic
                // if let Operand::Reg(reg) = operand {
                //     let reg = get_1byte_reg(reg);
                //     return writeln!(f, "\tset{}\t{}", condition, reg);
                // }
                writeln!(f, "\tset{}\t{}", condition, operand)
            }
            Instruction::Label(label) => writeln!(f, ".L{}:", label),
            Instruction::Push(operand) => {
                let operand = operand.as_64bit();
                writeln!(f, "\tpushq\t{}", operand)
            }
            Instruction::Pop(operand) => {
                let operand = operand.as_64bit();
                writeln!(f, "\tpopq\t{}", operand)
            }
            Instruction::Call(identifier) => {
                if cfg!(target_os = "linux") {
                    writeln!(f, "\tcall\t{}@PLT", identifier)
                } else {
                    writeln!(f, "\tcall\t_{}", identifier)
                }
            }
            Instruction::DeallocateStack(size) => writeln!(f, "\taddq\t${}, %rsp", size),
            Instruction::SpanOnly(span) => writeln!(
                f,
                "\n\t# {}",
                span.value(&source).split('\n').next().unwrap()
            ),
            ins => unimplemented!("{:?}", ins),
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

#[derive(Debug, PartialEq, Clone)]
pub enum Operand {
    Imm(i32),
    Reg(Reg),
    Reg8(Reg8),
    Reg16(Reg16),
    Reg32(Reg32),
    Reg64(Reg64),
    Pseudo(Identifier),
    Stack(i32),
    Data(Identifier),
}

impl Operand {
    fn is_memory(&self) -> bool {
        matches!(self, Operand::Stack(_) | Operand::Data(_))
    }

    fn as_8bit(&self) -> Operand {
        match self {
            Operand::Reg(reg) => Operand::Reg8(reg.as_8bit()),
            _ => self.clone(),
        }
    }

    fn as_16bit(&self) -> Operand {
        match self {
            Operand::Reg(reg) => Operand::Reg16(reg.as_16bit()),
            _ => self.clone(),
        }
    }

    fn as_32bit(&self) -> Operand {
        match self {
            Operand::Reg(reg) => Operand::Reg32(reg.as_32bit()),
            _ => self.clone(),
        }
    }

    fn as_64bit(&self) -> Operand {
        match self {
            Operand::Reg(reg) => Operand::Reg64(reg.as_64bit()),
            _ => self.clone(),
        }
    }
}

impl Display for Operand {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Operand::Imm(imm) => write!(f, "${}", imm),
            Operand::Reg(reg) => write!(f, "{}", reg),
            Operand::Reg8(reg) => write!(f, "{}", reg),
            Operand::Reg16(reg) => write!(f, "{}", reg),
            Operand::Reg32(reg) => write!(f, "{}", reg),
            Operand::Reg64(reg) => write!(f, "{}", reg),
            Operand::Pseudo(pseudo) => write!(f, "{}", pseudo),
            Operand::Stack(offset) => write!(f, "{}(%rbp)", offset),
            Operand::Data(identifier) => write!(f, "{identifier}(%rip)"),
        }
    }
}

impl From<ir::Val> for Operand {
    fn from(val: ir::Val) -> Self {
        match val {
            ir::Val::Constant(value, _span) => Operand::Imm(value),
            ir::Val::Var(identifier, _span) => Operand::Pseudo(identifier),
        }
    }
}

pub struct Assembler<'a> {
    source: &'a str,
}

impl<'a> Assembler<'a> {
    pub fn new(source: &'a str) -> Assembler {
        Assembler { source }
    }

    pub fn assemble(&self, program: ir::Program, symbols: SymbolMap) -> miette::Result<Program> {
        let program = Program {
            top_level: program.top_level.into_iter().map(|f| f.into()).collect(),
            symbols,
        };

        println!("\nRaw assembly:");
        for function in program.top_level.iter() {
            println!("{:?}", function);
        }

        let program = self.replace_pseudoregisters(program);
        let program = self.fixup_instructions(program);
        Ok(program)
    }

    pub fn replace_pseudoregisters(&self, program: Program) -> Program {
        let top_level = program
            .top_level
            .into_iter()
            .map(|top_level| match top_level {
                TopLevel::Function(function) => TopLevel::Function(
                    self.replace_pseudoregisters_function(function, &program.symbols),
                ),
                other => other,
            })
            .collect();

        Program {
            top_level,
            symbols: program.symbols,
        }
    }

    pub fn replace_pseudoregisters_function(
        &self,
        function: Function,
        symbols: &SymbolMap,
    ) -> Function {
        fn handle_operand(
            symbols: &SymbolMap,
            operand: Operand,
            stack_offset: &mut i32,
            stack_map: &mut HashMap<Identifier, i32>,
        ) -> Operand {
            match operand {
                Operand::Pseudo(pseudo) => {
                    let symbol = symbols.get(&pseudo);
                    if let Some(offset) = stack_map.get(&pseudo) {
                        Operand::Stack(*offset)
                    } else {
                        if let Some(symbol) = symbol {
                            if let TypeInfo::Variable(var) = &symbol.info {
                                if let VarAttrs::Static { global, .. } = &var.attrs {
                                    return Operand::Data(pseudo);
                                }
                            }
                        }

                        *stack_offset -= 4;
                        stack_map.insert(pseudo, *stack_offset);
                        Operand::Stack(*stack_offset)
                    }
                }
                _ => operand,
            }
        }

        let mut stack_size = 0;
        let mut pseudo_map = HashMap::new();

        let mut instructions = function
            .instructions
            .into_iter()
            .map(|instruction| match instruction {
                Instruction::Mov(src, dst) => {
                    let src = handle_operand(symbols, src, &mut stack_size, &mut pseudo_map);
                    let dst = handle_operand(symbols, dst, &mut stack_size, &mut pseudo_map);
                    Instruction::Mov(src, dst)
                }
                Instruction::Unary(op, operand) => {
                    let operand =
                        handle_operand(symbols, operand, &mut stack_size, &mut pseudo_map);
                    Instruction::Unary(op, operand)
                }
                Instruction::AllocateStack(size) => {
                    stack_size -= size;
                    Instruction::AllocateStack(size)
                }
                Instruction::Idiv(operand) => {
                    let operand =
                        handle_operand(symbols, operand, &mut stack_size, &mut pseudo_map);
                    Instruction::Idiv(operand)
                }
                Instruction::Cdq => Instruction::Cdq,
                Instruction::Binary(op, src1, src2) => {
                    let src1 = handle_operand(symbols, src1, &mut stack_size, &mut pseudo_map);
                    let src2 = handle_operand(symbols, src2, &mut stack_size, &mut pseudo_map);
                    Instruction::Binary(op, src1, src2)
                }
                Instruction::Cmp(op1, op2) => {
                    let op1 = handle_operand(symbols, op1, &mut stack_size, &mut pseudo_map);
                    let op2 = handle_operand(symbols, op2, &mut stack_size, &mut pseudo_map);
                    Instruction::Cmp(op1, op2)
                }
                Instruction::SetCC(condition, operand) => {
                    let operand =
                        handle_operand(symbols, operand, &mut stack_size, &mut pseudo_map);
                    Instruction::SetCC(condition, operand)
                }
                Instruction::Ret(ret) => Instruction::Ret(ret),
                Instruction::Push(operand) => {
                    let operand =
                        handle_operand(symbols, operand, &mut stack_size, &mut pseudo_map);
                    Instruction::Push(operand)
                }
                _ => instruction,
            })
            .collect::<Vec<_>>();

        if stack_size != 0 {
            let aligned_stack_size = (-stack_size + 15) & !15;
            instructions.insert(0, Instruction::AllocateStack(aligned_stack_size));
        }

        Function {
            name: function.name,
            instructions,
            global: function.global,
            stack_size,
        }
    }

    pub fn fixup_instructions(&self, program: Program) -> Program {
        let function_definitions = program
            .top_level
            .into_iter()
            .map(|top_level| match top_level {
                TopLevel::Function(function) => {
                    TopLevel::Function(self.fixup_instructions_function(function))
                }
                other => other,
            })
            .collect();

        Program {
            top_level: function_definitions,
            symbols: program.symbols,
        }
    }

    pub fn fixup_instructions_function(&self, function: Function) -> Function {
        let instructions = function
            .instructions
            .into_iter()
            .flat_map(|instruction| {
                println!("instr: {instruction:?}");
                match instruction {
                    Instruction::Mov(src, dst) => match (src, dst) {
                        (Operand::Stack(src), Operand::Stack(dst)) => {
                            vec![
                                Instruction::Mov(Operand::Stack(src), Operand::Reg(Reg::R10)),
                                Instruction::Mov(Operand::Reg(Reg::R10), Operand::Stack(dst)),
                            ]
                        }
                        (Operand::Data(src), Operand::Stack(dst)) => {
                            vec![
                                Instruction::Mov(Operand::Data(src), Operand::Reg(Reg::R10)),
                                Instruction::Mov(Operand::Reg(Reg::R10), Operand::Stack(dst)),
                            ]
                        }
                        (Operand::Stack(src), Operand::Data(dst)) => {
                            vec![
                                Instruction::Mov(Operand::Stack(src), Operand::Reg(Reg::R10)),
                                Instruction::Mov(Operand::Reg(Reg::R10), Operand::Data(dst)),
                            ]
                        }
                        (src, dst) => vec![Instruction::Mov(src, dst)],
                    },
                    Instruction::Idiv(operand) => vec![
                        Instruction::Mov(operand, Operand::Reg(Reg::R10)),
                        Instruction::Idiv(Operand::Reg(Reg::R10)),
                    ],
                    Instruction::Binary(op, src1, src2) => match (&op, src1, src2) {
                        (BinaryOperator::Add, src1, src2)
                        | (BinaryOperator::Sub, src1, src2)
                        | (BinaryOperator::ShiftLeft, src1, src2)
                        | (BinaryOperator::ShiftRight, src1, src2) => {
                            if src1.is_memory() || src2.is_memory() {
                                vec![
                                    Instruction::Mov(src1, Operand::Reg(Reg::R10)),
                                    Instruction::Binary(op, Operand::Reg(Reg::R10), src2),
                                ]
                            } else {
                                vec![Instruction::Binary(op, src1, src2)]
                            }
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
                    Instruction::Cmp(op1, Operand::Imm(op2)) => vec![
                        Instruction::Mov(Operand::Imm(op2), Operand::Reg(Reg::R11)),
                        Instruction::Cmp(op1, Operand::Reg(Reg::R11)),
                    ],
                    Instruction::Cmp(op1, op2) => {
                        if op1.is_memory() || op2.is_memory() {
                            vec![
                                Instruction::Mov(op1, Operand::Reg(Reg::R10)),
                                Instruction::Cmp(Operand::Reg(Reg::R10), op2),
                            ]
                        } else {
                            vec![Instruction::Cmp(op1, op2)]
                        }
                    }

                    instr => vec![instr],
                }
            })
            .collect::<Vec<_>>();

        Function {
            name: function.name,
            global: function.global,
            instructions,
            stack_size: function.stack_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use ir::Ir;

    use crate::{lexer::Lexer, parser::Parser};

    use super::*;

    #[test]
    fn test_fun_call() {
        // let code = r#"
        //     int putchar(int c);
        //     int main(void) {
        //         putchar(72);
        //     }
        // "#;
        //
        // let mut lexer = Lexer::new(code);
        // let tokens = lexer.run().unwrap();
        // let mut parser = Parser::new(code, &tokens);
        // let ast = parser.run().unwrap();
        // let ir = Ir::new(ast).run().unwrap();
        // let assembler = Assembler::new(ir);
        // let program = assembler.assemble().unwrap();
        // println!("{}", program);
    }

    //     #[test]
    //     fn test_assembly() {
    //         let ir = ir::Program {
    //             function_definition: ir::FuncDefinition {
    //                 name: "main".to_string(),
    //                 instructions: vec![
    //                     ir::Instruction::Unary(
    //                         ir::UnaryOperator::Negate,
    //                         ir::Val::Constant(8),
    //                         ir::Val::Var("tmp.0".to_string()),
    //                     ),
    //                     ir::Instruction::Unary(
    //                         ir::UnaryOperator::Complement,
    //                         ir::Val::Var("tmp.0".to_string()),
    //                         ir::Val::Var("tmp.1".to_string()),
    //                     ),
    //                     ir::Instruction::Unary(
    //                         ir::UnaryOperator::Negate,
    //                         ir::Val::Var("tmp.1".to_string()),
    //                         ir::Val::Var("tmp.2".to_string()),
    //                     ),
    //                     ir::Instruction::Return(ir::Val::Var("tmp.2".to_string())),
    //                 ],
    //             },
    //         };
    //
    //         let assembler = Assembler::new(ir);
    //         let program = assembler.run().unwrap();
    //         println!("{}", program);
    //         // assert_eq!(
    //         //     program.function_definition,
    //         //     Function {
    //         //         name: "main".to_string(),
    //         //         instructions: vec![
    //         //             Instruction::AllocateStack(12),
    //         //             Instruction::Mov(Operand::Imm(8), Operand::Stack(-4)),
    //         //             Instruction::Unary(UnaryOperator::Neg, Operand::Stack(-4)),
    //         //             Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
    //         //             Instruction::Unary(UnaryOperator::Not, Operand::Stack(-8)),
    //         //             Instruction::Mov(Operand::Stack(-8), Operand::Stack(-12)),
    //         //             Instruction::Unary(UnaryOperator::Neg, Operand::Stack(-12)),
    //         //             Instruction::Mov(Operand::Stack(-12), Operand::Reg(Reg::AX)),
    //         //             Instruction::Ret(Ret),
    //         //         ]
    //         //     }
    //         // );
    //     }
    //
    //     #[test]
    //     fn test_from_ret() {
    //         let ir = ir::Program {
    //             function_definition: ir::FuncDefinition {
    //                 name: "main".to_string(),
    //                 instructions: vec![ir::Instruction::Return(ir::Val::Constant(42))],
    //             },
    //         };
    //
    //         let program = Program::from(ir);
    //         assert_eq!(
    //             program.function_definition,
    //             Function {
    //                 name: "main".to_string(),
    //                 instructions: vec![
    //                     Instruction::Mov(Operand::Imm(42), Operand::Reg(Reg::AX)),
    //                     Instruction::Ret(Ret)
    //                 ]
    //             }
    //         );
    //     }
    //
    //     #[test]
    //     fn test_from_unary() {
    //         let ir = ir::Program {
    //             function_definition: ir::FuncDefinition {
    //                 name: "main".to_string(),
    //                 instructions: vec![ir::Instruction::Unary(
    //                     ir::UnaryOperator::Negate,
    //                     ir::Val::Constant(42),
    //                     ir::Val::Var("tmp.0".to_string()),
    //                 )],
    //             },
    //         };
    //
    //         let program = Program::from(ir);
    //         assert_eq!(
    //             program.function_definition,
    //             Function {
    //                 name: "main".to_string(),
    //                 instructions: vec![
    //                     Instruction::Mov(Operand::Imm(42), Operand::Pseudo("tmp.0".to_string())),
    //                     Instruction::Unary(UnaryOperator::Neg, Operand::Pseudo("tmp.0".to_string())),
    //                 ]
    //             }
    //         );
    //     }
    //
    //     fn ir_prog(instrs: Vec<ir::Instruction>) -> ir::Program {
    //         ir::Program {
    //             function_definition: ir::FuncDefinition {
    //                 name: "main".to_string(),
    //                 instructions: instrs,
    //             },
    //         }
    //     }
    //
    //     #[test]
    //     fn test_pseudo_and_vars() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::And,
    //             ir::Val::Var("tmp.0".to_string()),
    //             ir::Val::Var("tmp.1".to_string()),
    //             ir::Val::Var("tmp.2".to_string()),
    //         )]);
    //
    //         let asm = Assembler::new(prog.clone());
    //         let prog = asm.replace_pseudoregisters(prog.into());
    //
    //         let expected = vec![
    //             Instruction::AllocateStack(12),
    //             Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
    //             Instruction::Binary(BinaryOperator::And, Operand::Stack(-12), Operand::Stack(-8)),
    //         ];
    //
    //         assert_eq!(prog.function_definition.instructions, expected);
    //     }
    //
    //     #[test]
    //     fn test_fixup_and() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::And,
    //             ir::Val::Var("tmp.0".to_string()),
    //             ir::Val::Var("tmp.1".to_string()),
    //             ir::Val::Var("tmp.2".to_string()),
    //         )]);
    //
    //         let asm = Assembler::new(prog.clone());
    //         let prog = asm.replace_pseudoregisters(prog.into());
    //
    //         let expected = vec![
    //             Instruction::AllocateStack(12),
    //             Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
    //             Instruction::Binary(BinaryOperator::And, Operand::Stack(-12), Operand::Stack(-8)),
    //         ];
    //
    //         assert_eq!(prog.function_definition.instructions, expected);
    //     }
    //     #[test]
    //     fn test_pseudo_mul_vars() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::Multiply,
    //             ir::Val::Var("tmp.0".to_string()),
    //             ir::Val::Var("tmp.1".to_string()),
    //             ir::Val::Var("tmp.2".to_string()),
    //         )]);
    //
    //         let asm = Assembler::new(prog.clone());
    //         let prog = asm.replace_pseudoregisters(prog.into());
    //
    //         let expected = vec![
    //             Instruction::AllocateStack(12),
    //             Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
    //             Instruction::Binary(BinaryOperator::Mul, Operand::Stack(-12), Operand::Stack(-8)),
    //         ];
    //
    //         assert_eq!(prog.function_definition.instructions, expected);
    //     }
    //
    //     #[test]
    //     fn test_pseudo_mul_const_var() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::Multiply,
    //             ir::Val::Constant(3),
    //             ir::Val::Var("tmp.1".to_string()),
    //             ir::Val::Var("tmp.2".to_string()),
    //         )]);
    //
    //         let asm = Assembler::new(prog.clone());
    //         let prog = asm.replace_pseudoregisters(prog.into());
    //
    //         let expected = vec![
    //             Instruction::AllocateStack(8),
    //             Instruction::Mov(Operand::Imm(3), Operand::Stack(-4)),
    //             Instruction::Binary(BinaryOperator::Mul, Operand::Stack(-8), Operand::Stack(-4)),
    //         ];
    //
    //         assert_eq!(prog.function_definition.instructions, expected);
    //
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::Multiply,
    //             ir::Val::Var("tmp.1".to_string()),
    //             ir::Val::Constant(3),
    //             ir::Val::Var("tmp.2".to_string()),
    //         )]);
    //
    //         let asm = Assembler::new(prog.clone());
    //         let prog = asm.replace_pseudoregisters(prog.into());
    //
    //         let expected = vec![
    //             Instruction::AllocateStack(8),
    //             Instruction::Mov(Operand::Stack(-4), Operand::Stack(-8)),
    //             Instruction::Binary(BinaryOperator::Mul, Operand::Imm(3), Operand::Stack(-8)),
    //         ];
    //
    //         assert_eq!(prog.function_definition.instructions, expected);
    //     }
    //
    //     #[test]
    //     fn test_imull_gen() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::Multiply,
    //             ir::Val::Constant(3),
    //             ir::Val::Var("tmp.0".to_string()),
    //             ir::Val::Var("tmp.1".to_string()),
    //         )]);
    //
    //         let assembler = Assembler::new(prog.clone());
    //         let program = assembler.run().unwrap();
    //         println!("{:#?}", program);
    //         println!("{}", program);
    //         // assert_eq!(
    //         //     program.function_definition,
    //         //     Function {
    //         //         name: "main".to_string(),
    //         //         instructions: vec![
    //         //             Instruction::Mov(Operand::Imm(42), Operand::Pseudo("tmp.0".to_string())),
    //         //             Instruction::Unary(UnaryOperator::Neg, Operand::Pseudo("tmp.0".to_string())),
    //         //         ]
    //         //     }
    //         // );
    //     }
    //
    //     #[test]
    //     fn test_andl_gen() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::And,
    //             ir::Val::Constant(3),
    //             ir::Val::Var("tmp.0".to_string()),
    //             ir::Val::Var("tmp.1".to_string()),
    //         )]);
    //
    //         let assembler = Assembler::new(prog.clone());
    //         let program = assembler.run().unwrap();
    //         println!("{:#?}", program);
    //         println!("{}", program);
    //     }
    //
    //     #[test]
    //     fn test_orl_gen() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::Or,
    //             ir::Val::Constant(3),
    //             ir::Val::Var("tmp.0".to_string()),
    //             ir::Val::Var("tmp.1".to_string()),
    //         )]);
    //
    //         let assembler = Assembler::new(prog.clone());
    //         let program = assembler.run().unwrap();
    //         println!("{:#?}", program);
    //         println!("{}", program);
    //     }
    //
    //     #[test]
    //     fn test_xorl_gen() {
    //         let prog = ir_prog(vec![ir::Instruction::Binary(
    //             ir::BinaryOperator::BitwiseXor,
    //             ir::Val::Constant(3),
    //             ir::Val::Var("tmp.0".to_string()),
    //             ir::Val::Var("tmp.1".to_string()),
    //         )]);
    //
    //         let assembler = Assembler::new(prog.clone());
    //         let program = assembler.run().unwrap();
    //         println!("{:#?}", program);
    //         println!("{}", program);
    //     }
}
