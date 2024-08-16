#![allow(unused)]
use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
pub struct Program {
    function_definition: Function,
}

impl From<crate::parser::Program> for Program {
    fn from(program: crate::parser::Program) -> Self {
        Program {
            function_definition: program.function_definition.into(),
        }
    }
}

#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
}

impl From<crate::parser::Function> for Function {
    fn from(function: crate::parser::Function) -> Self {
        Function {
            name: function.name,
            instructions: function.body.into(),
        }
    }
}

#[derive(Debug)]
pub struct Mov {
    pub exp: Exp,
    pub reg: Register,
}

impl Display for Mov {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "mov {}, {}", self.exp, self.reg)
    }
}

#[derive(Debug)]
pub struct Ret;

impl Display for Ret {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "ret")
    }
}

#[derive(Debug)]
pub struct Imm {
    pub value: i32,
}

impl Display for Imm {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Debug)]
pub struct Register;

impl Display for Register {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "eax")
    }
}

#[derive(Debug)]
pub enum Instruction {
    Mov(Mov),
    Ret(Ret),
}

impl From<crate::parser::Statement> for Vec<Instruction> {
    fn from(statement: crate::parser::Statement) -> Self {
        vec![
            Instruction::Mov(Mov {
                exp: statement.return_exp.into(),
                reg: Register,
            }),
            Instruction::Ret(Ret),
        ]
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Instruction::Mov(mov) => write!(f, "{}", mov),
            Instruction::Ret(ret) => write!(f, "{}", ret),
        }
    }
}

#[derive(Debug)]
pub enum Exp {
    Imm(Imm),
    Register,
}

impl Display for Exp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Exp::Imm(imm) => write!(f, "{}", imm),
            Exp::Register => write!(f, "eax"),
        }
    }
}

impl From<crate::parser::Exp> for Exp {
    fn from(exp: crate::parser::Exp) -> Self {
        Exp::Imm(Imm {
            value: exp.constant,
        })
    }
}

pub struct Assembler {
    program: crate::parser::Program,
}

impl Assembler {
    pub fn new(program: crate::parser::Program) -> Assembler {
        Assembler { program }
    }

    pub fn run(self) -> anyhow::Result<Program> {
        Ok(self.program.into())
    }
}
