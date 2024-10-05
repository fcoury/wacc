use std::sync::atomic::{AtomicI32, Ordering};

use crate::{
    lexer::Span,
    parser::{self, BlockItem, Identifier},
    semantic::{InitialValue, SymbolMap, TypeInfo, VarAttrs},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub top_level: Vec<TopLevel>,
}

impl Program {
    pub fn iter(&self) -> std::slice::Iter<TopLevel> {
        self.top_level.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopLevel {
    Function(Function),
    StaticVariable(StaticVar),
}

impl TryFrom<parser::Program> for Program {
    type Error = miette::Error;

    fn try_from(program: parser::Program) -> miette::Result<Self> {
        let functions = program
            .declarations
            .into_iter()
            .filter_map(|d| match d {
                parser::Declaration::Function(function_decl, _) => Some(function_decl),
                _ => None,
            })
            .filter(|fd| fd.body.is_some())
            .map(|d| d.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let top_level = functions.into_iter().map(TopLevel::Function).collect();

        Ok(Program { top_level })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticVar {
    pub name: Identifier,
    pub global: bool,
    pub init: Val,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Identifier,
    pub global: bool,
    pub params: Vec<Identifier>,
    pub instructions: Vec<Instruction>,
}

impl TryFrom<parser::FunctionDecl> for Function {
    type Error = miette::Error;

    fn try_from(function: parser::FunctionDecl) -> miette::Result<Self> {
        let mut context = Context::new();
        Ok(Function {
            name: function.name.clone(),
            global: true,
            params: function
                .params
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>(),
            instructions: function.into_instructions(&mut context)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    SpanOnly(Span),
    Return(Val, Option<Span>),
    Unary(UnaryOperator, Val, Val, Option<Span>),
    Binary(BinaryOperator, Val, Val, Val, Option<Span>),
    Copy(Val, Val, Option<Span>),
    Jump(Identifier, Option<Span>),
    JumpIfZero(Val, Identifier, Option<Span>),
    JumpIfNotZero(Val, Identifier, Option<Span>),
    Label(Identifier, Option<Span>),
    FunCall(Identifier, Vec<Val>, Val, Option<Span>),
}

trait IntoInstructions {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>>;
}

impl IntoInstructions for parser::FunctionDecl {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>> {
        println!("handling fn: {:?}", self);
        let Some(body) = self.body else {
            return Ok(vec![]);
        };

        let mut instructions = body.into_instructions(context)?;
        // if no return is found
        if let Some(last) = instructions.last() {
            if !matches!(last, Instruction::Return(..)) {
                instructions.push(Instruction::Return(Val::Constant(0, None), None));
            }
        } else {
            instructions.push(Instruction::Return(Val::Constant(0, None), None));
        }

        Ok(instructions)
    }
}

impl IntoInstructions for parser::Block {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>> {
        Ok(self
            .into_iter()
            .map(|item| match item {
                BlockItem::Statement(statement, _span) => {
                    statement.clone().into_instructions(context)
                }
                BlockItem::Declaration(declaration, _span) => {
                    declaration.clone().into_instructions(context)
                }
            })
            .collect::<Result<Vec<Vec<Instruction>>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>())
    }
}

impl IntoInstructions for parser::Exp {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>> {
        let mut instructions = vec![];
        emit_ir(self, &mut instructions, context);
        Ok(instructions)
    }
}

impl IntoInstructions for parser::ForInit {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>> {
        match self {
            parser::ForInit::Declaration(decl, _span) => decl.into_instructions(context),
            parser::ForInit::Expression(exp, _span) => {
                let mut instructions = vec![];

                if let Some(exp) = exp {
                    // we disregard any returns of the init expression, but generate the
                    // instructions
                    _ = emit_ir(exp, &mut instructions, context);
                }

                Ok(instructions)
            }
        }
    }
}

impl IntoInstructions for parser::VarDecl {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>> {
        let Some(init) = self.init else {
            return Ok(vec![]);
        };

        // static variables are initialized only once
        if self.storage_class == Some(parser::StorageClass::Static) {
            return Ok(vec![]);
        }

        let mut instructions = vec![];
        emit_assignment(self.name, init, &mut instructions, context, self.span);
        Ok(instructions)
    }
}

fn break_label_for(label: String) -> Identifier {
    Identifier::new(format!("break_{label}"))
}

fn continue_label_for(label: String) -> Identifier {
    Identifier::new(format!("continue_{label}"))
}

impl IntoInstructions for parser::Statement {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>> {
        let mut instructions = vec![Instruction::SpanOnly(self.span())];

        match self {
            parser::Statement::Return(exp, span) => {
                let val = emit_ir(exp, &mut instructions, context);
                instructions.push(Instruction::Return(val, Some(span)));
            }
            parser::Statement::Expression(exp, _span) => {
                // instructions.push(Instruction::SpanOnly(span));
                emit_ir(exp, &mut instructions, context);
            }
            parser::Statement::Null => {}
            parser::Statement::If(cond, then, else_, _span) => {
                // instructions.push(Instruction::SpanOnly(span));
                let cond = emit_ir(cond, &mut instructions, context);
                let end_label = context.next_label("end");
                let else_label = context.next_label("else");

                instructions.push(Instruction::JumpIfZero(
                    cond.clone(),
                    else_label.clone(),
                    None,
                ));
                let then_instructions = then.into_instructions(context)?;
                instructions.extend(then_instructions);
                instructions.push(Instruction::Jump(end_label.clone(), None));
                instructions.push(Instruction::Label(else_label.clone(), None));
                if let Some(else_) = else_ {
                    let else_instructions = else_.into_instructions(context)?;
                    instructions.extend(else_instructions);
                }
                instructions.push(Instruction::Label(end_label.clone(), None));
            }
            parser::Statement::Compound(block, _span) => {
                instructions.extend(block.into_instructions(context)?)
            }
            parser::Statement::Break(label, span) => {
                let Some(label) = label else {
                    miette::bail!("Break statement outside loop");
                };

                instructions.push(Instruction::Jump(break_label_for(label), Some(span)));
            }
            parser::Statement::Continue(label, span) => {
                let Some(label) = label else {
                    miette::bail!("Continue statement outside loop");
                };

                instructions.push(Instruction::Jump(continue_label_for(label), Some(span)))
            }
            parser::Statement::While {
                condition,
                body,
                label,
                span: _,
            } => {
                let Some(label) = label else {
                    miette::bail!("Unexpected error: While label is missing");
                };
                let continue_label = continue_label_for(label.clone());
                let break_label = break_label_for(label.clone());

                // start label
                instructions.push(Instruction::Label(continue_label.clone(), None));

                // condition
                let cond = emit_ir(condition, &mut instructions, context);
                instructions.push(Instruction::JumpIfZero(
                    cond.clone(),
                    break_label.clone(),
                    None,
                ));

                // body instructions
                let body_instructions = body.into_instructions(context)?;
                instructions.extend(body_instructions);

                // jump back to continue label
                instructions.push(Instruction::Jump(continue_label, None));

                // break label
                instructions.push(Instruction::Label(break_label, None));
            }
            parser::Statement::DoWhile {
                body,
                condition,
                label,
                span: _,
            } => {
                let Some(start_label) = label else {
                    miette::bail!("Unexpected error: DoWhile label is missing");
                };
                let continue_label = continue_label_for(start_label.clone());
                let break_label = break_label_for(start_label.clone());
                let start_label = Identifier::new(start_label.clone());

                // instructions.push(Instruction::SpanOnly(span));

                // start label
                instructions.push(Instruction::Label(start_label.clone(), None));

                // body instructions
                let body_instructions = body.into_instructions(context)?;
                instructions.extend(body_instructions);

                // continue label
                instructions.push(Instruction::Label(continue_label, None));

                // condition
                let cond = emit_ir(condition, &mut instructions, context);
                instructions.push(Instruction::JumpIfNotZero(cond.clone(), start_label, None));

                // break label
                instructions.push(Instruction::Label(break_label, None));
            }
            parser::Statement::For {
                init,
                condition,
                post,
                body,
                label,
                span: _,
            } => {
                let Some(start_label) = label else {
                    miette::bail!("Unexpected error: For label is missing");
                };
                let break_label = break_label_for(start_label.clone());
                let continue_label = continue_label_for(start_label.clone());
                let start_label = Identifier::new(start_label.clone());

                // instructions.push(Instruction::SpanOnly(span));

                if let Some(init) = init {
                    instructions.extend(init.into_instructions(context)?);
                }

                // start label
                instructions.push(Instruction::Label(start_label.clone(), None));

                // condition
                if let Some(condition) = condition {
                    let cond = emit_ir(condition, &mut instructions, context);
                    instructions.push(Instruction::JumpIfZero(
                        cond.clone(),
                        break_label.clone(),
                        None,
                    ));
                } else {
                    // if condition is absent, C standard says this expression is "replaced by a
                    // nonzero constant" (section 6.8.5.3, paragraph 2)
                    instructions.push(Instruction::JumpIfZero(
                        Val::Constant(1, None),
                        break_label.clone(),
                        None,
                    ));
                }

                // body instructions
                instructions.extend(body.into_instructions(context)?);

                // continue label
                instructions.push(Instruction::Label(continue_label.clone(), None));

                // post
                if let Some(post) = post {
                    instructions.extend(post.into_instructions(context)?);
                }

                // jumps to start
                instructions.push(Instruction::Jump(start_label, None));

                // break label
                instructions.push(Instruction::Label(break_label, None));
            }
        };

        Ok(instructions)
    }
}

impl IntoInstructions for parser::Declaration {
    fn into_instructions(self, context: &mut Context) -> miette::Result<Vec<Instruction>> {
        match self {
            parser::Declaration::Var(decl, _span) => decl.into_instructions(context),
            parser::Declaration::Function(decl, _span) => decl.into_instructions(context),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Constant(i32, Option<Span>),
    Var(Identifier, Option<Span>),
}

impl Val {
    pub fn span(&self) -> Option<Span> {
        match self {
            Val::Constant(_, span) => *span,
            Val::Var(_, span) => *span,
        }
    }
}

pub fn emit_assignment(
    name: Identifier,
    exp: parser::Exp,
    instructions: &mut Vec<Instruction>,
    context: &mut Context,
    span: Span,
) -> Val {
    let val = emit_ir(exp, instructions, context);
    instructions.push(Instruction::SpanOnly(span));
    instructions.push(Instruction::Copy(
        val.clone(),
        Val::Var(name.clone(), name.span()),
        val.clone().span(),
    ));
    val
}

pub fn emit_ir(
    exp: parser::Exp,
    instructions: &mut Vec<Instruction>,
    context: &mut Context,
) -> Val {
    match exp {
        parser::Exp::Var(name, _typ, span) => Val::Var(name, Some(span)),
        parser::Exp::Assignment(exp, rhs, _typ, span) => match exp.as_ref() {
            parser::Exp::Var(name, _typ, _span) => {
                emit_assignment(name.clone(), *rhs, instructions, context, span)
            }
            parser::Exp::Constant(_, _typ, _) => todo!(),
            parser::Exp::Assignment(_, _, _typ, _) => todo!(),
            parser::Exp::Unary(_, _, _typ, _) => todo!(),
            parser::Exp::BinaryOperation(_, _, _, _typ, _) => todo!(),
            parser::Exp::Conditional(_, _, _, _typ, _) => todo!(),
            parser::Exp::FunctionCall(_, _, _typ, _) => todo!(),
            parser::Exp::Cast(_, _exp, _typ, _span) => todo!(),
        },
        parser::Exp::Constant(_value, _typ, _span) => todo!(), // Val::Constant(value, Some(span)),
        parser::Exp::Unary(operator, exp, _typ, span) => {
            let src = emit_ir(*exp, instructions, context);
            let dst = Val::Var(context.next_var(), None);
            instructions.push(Instruction::Unary(
                operator.into(),
                src,
                dst.clone(),
                Some(span),
            ));
            dst
        }
        parser::Exp::BinaryOperation(oper, v1, v2, _typ, _span) => match oper {
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
                    None,
                ));

                // Evaluate second operand
                let val2 = emit_ir(*v2, instructions, context);

                // if true (not 0), short-circuit
                instructions.push(Instruction::JumpIfNotZero(
                    val2.clone(),
                    short_circuit_label.clone(),
                    None,
                ));

                // Set result based on second operand
                instructions.push(Instruction::Copy(
                    Val::Constant(0, None),
                    Val::Var(result.clone(), None),
                    None,
                ));
                // And jump to end
                instructions.push(Instruction::Jump(end_label.clone(), None));

                // False label
                instructions.push(Instruction::Label(short_circuit_label.clone(), None));
                instructions.push(Instruction::Copy(
                    Val::Constant(1, None),
                    Val::Var(result.clone(), None),
                    None,
                ));

                // End label
                instructions.push(Instruction::Label(end_label.clone(), None));

                Val::Var(result, None)
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
                    None,
                ));

                // Evaluate second operand
                let val2 = emit_ir(*v2, instructions, context);

                // For AND: if false (0), short-circuit
                instructions.push(Instruction::JumpIfZero(
                    val2.clone(),
                    short_circuit_label.clone(),
                    None,
                ));

                // Set result based on second operand
                instructions.push(Instruction::Copy(
                    Val::Constant(1, None),
                    Val::Var(result.clone(), None),
                    None,
                ));
                // And jump to end
                instructions.push(Instruction::Jump(end_label.clone(), None));

                // False label
                instructions.push(Instruction::Label(short_circuit_label.clone(), None));
                instructions.push(Instruction::Copy(
                    Val::Constant(0, None),
                    Val::Var(result.clone(), None),
                    None,
                ));

                // End label
                instructions.push(Instruction::Label(end_label.clone(), None));

                Val::Var(result, None)
            }
            _ => {
                let val1 = emit_ir(*v1, instructions, context);
                let val2 = emit_ir(*v2, instructions, context);
                let dst = Val::Var(context.next_var(), None);
                instructions.push(Instruction::Binary(
                    oper.into(),
                    val1,
                    val2,
                    dst.clone(),
                    None,
                ));
                dst
            }
        },
        parser::Exp::Conditional(cond, e1, e2, _typ, _span) => {
            // Emit instructions for condition
            let c = emit_ir(*cond, instructions, context);

            // Create labels
            let e2_label = context.next_label("else");
            let end_label = context.next_label("end");

            // Create result variable
            let result = context.next_var();

            // Jump to e2 if condition is zero
            instructions.push(Instruction::JumpIfZero(c, e2_label.clone(), None));

            // Emit instructions for e1
            let v1 = emit_ir(*e1, instructions, context);

            // Assign result of e1 to result
            instructions.push(Instruction::Copy(v1, Val::Var(result.clone(), None), None));

            // Jump to end
            instructions.push(Instruction::Jump(end_label.clone(), None));

            // Label for e2
            instructions.push(Instruction::Label(e2_label, None));

            // Emit instructions for e2
            let v2 = emit_ir(*e2, instructions, context);

            // Assign result of e2 to result
            instructions.push(Instruction::Copy(v2, Val::Var(result.clone(), None), None));

            // End label
            instructions.push(Instruction::Label(end_label, None));

            // Return result
            Val::Var(result, None)
        }
        parser::Exp::FunctionCall(name, params, _typ, _span) => {
            let params = params
                .iter()
                .map(|p| emit_ir(p.clone(), instructions, context))
                .collect::<Vec<_>>();

            let result = context.next_var();

            instructions.push(Instruction::FunCall(
                name,
                params,
                Val::Var(result.clone(), None),
                None,
            ));

            Val::Var(result, None)
        }
        parser::Exp::Cast(_, _exp, _typ, _span) => todo!(),
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

    pub fn run(self, symbols: &SymbolMap) -> miette::Result<Program> {
        let mut program: Program = self.program.try_into()?;
        program.top_level.extend(convert_symbols_to_tacky(symbols));
        Ok(program)
    }
}

fn convert_symbols_to_tacky(symbols: &SymbolMap) -> Vec<TopLevel> {
    let mut tacky_defs = Vec::new();

    for (name, entry) in symbols.iter() {
        if let TypeInfo::Variable(var) = &entry.info {
            if let VarAttrs::Static { init, global } = &var.attrs {
                match init {
                    InitialValue::Initial(i) => {
                        tacky_defs.push(TopLevel::StaticVariable(StaticVar {
                            name: name.clone(),
                            global: *global,
                            // TODO: init: Val::Constant(*i, None),
                            init: todo!(),
                        }));
                    }
                    InitialValue::Tentative => {
                        tacky_defs.push(TopLevel::StaticVariable(StaticVar {
                            name: name.clone(),
                            global: *global,
                            init: Val::Constant(0, None),
                        }));
                    }
                    _ => {}
                }
            }
        }
    }

    tacky_defs
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
        Identifier::new(format!("tmp.{}", id))
    }

    pub fn next_label(&self, descr: &str) -> Identifier {
        let id = self.next_temp.fetch_add(1, Ordering::SeqCst);
        Identifier::new(format!("{descr}.label.{}", id))
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::{
    //     lexer::Lexer,
    //     parser::{self, Block, BlockItem, Parser},
    // };

    // #[test]
    // fn test_constant() {
    //     let program = parser::Program {
    //         declarations: vec![parser::Declaration::Function(parser::FunctionDecl {
    //             name: "main".to_string(),
    //             params: vec![],
    //             body: Some(Block {
    //                 items: vec![BlockItem::Statement(parser::Statement::Return(
    //                     parser::Exp::Constant(3),
    //                 ))],
    //             }),
    //             storage_class: None,
    //         })],
    //     };
    //     let program = Ir::new(program).run().unwrap();
    //     let instr = program
    //         .top_level
    //         .into_iter()
    //         .flat_map(|fd| fd.instructions)
    //         .collect::<Vec<_>>();
    //
    //     assert_eq!(
    //         instr.first().unwrap(),
    //         &Instruction::Return(Val::Constant(3))
    //     );
    // }

    #[test]
    fn test_unary_var_var() {
        // let program = parser::Program {
        //     declarations: vec![parser::Declaration::Function(parser::FunctionDecl {
        //         name: "main".to_string(),
        //         params: vec![],
        //         body: Some(Block {
        //             items: vec![BlockItem::Statement(parser::Statement::Return(
        //                 parser::Exp::Unary(
        //                     parser::UnaryOperator::Negate,
        //                     Box::new(parser::Exp::Var("a".to_string())),
        //                 ),
        //             ))],
        //         }),
        //         storage_class: None,
        //     })],
        // };
        //
        // let program = Ir::new(program).run().unwrap();
        // let instr = program
        //     .top_level
        //     .into_iter()
        //     .flat_map(|fd| fd.instructions)
        //     .collect::<Vec<_>>();
        //
        // println!("{:?}", instr);
    }

    #[test]
    fn test_unary() {
        // let input = "int main(void) { return ~2; }";
        // let mut lexer = Lexer::new(input);
        // let tokens = lexer.run().unwrap();
        // let mut parser = Parser::new(input, &tokens);
        // let ast = parser.run().unwrap();
        //
        // let program = Ir::new(ast).run().unwrap();
        // let instr = program
        //     .top_level
        //     .into_iter()
        //     .flat_map(|fd| fd.instructions)
        //     .collect::<Vec<_>>();
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
        // let input = "int main(void) { return -(~(-8)); }";
        // let mut lexer = Lexer::new(input);
        // let tokens = lexer.run().unwrap();
        // let mut parser = Parser::new(input, &tokens);
        // let ast = parser.run().unwrap();
        //
        // let program = Ir::new(ast).run().unwrap();
        // let instr = program
        //     .top_level
        //     .into_iter()
        //     .flat_map(|fd| fd.instructions)
        //     .collect::<Vec<_>>();
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
        // let input = "int main(void) { return 1 || 2; }";
        // let mut lexer = Lexer::new(input);
        // let tokens = lexer.run().unwrap();
        // let mut parser = Parser::new(input, &tokens);
        // let ast = parser.run().unwrap();
        //
        // let program = Ir::new(ast).run().unwrap();
        // let instr = program
        //     .top_level
        //     .into_iter()
        //     .flat_map(|fd| fd.instructions)
        //     .collect::<Vec<_>>();
        //
        // assert_eq!(
        //     instr,
        //     vec![
        //         Instruction::JumpIfNotZero(Val::Constant(1), "short_circuit.label.0".to_string()),
        //         Instruction::JumpIfNotZero(Val::Constant(2), "short_circuit.label.0".to_string()),
        //         Instruction::Copy(Val::Constant(0), Val::Var("tmp.2".to_string())),
        //         Instruction::Jump("end.label.1".to_string()),
        //         Instruction::Label("short_circuit.label.0".to_string()),
        //         Instruction::Copy(Val::Constant(1), Val::Var("tmp.2".to_string())),
        //         Instruction::Label("end.label.1".to_string()),
        //         Instruction::Return(Val::Var("tmp.2".to_string()))
        //     ]
        // );
    }
}
