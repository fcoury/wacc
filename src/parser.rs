#![allow(dead_code)]
use std::fmt;

use miette::{Diagnostic, LabeledSpan, SourceCode};
use strum::EnumProperty;
use strum_macros::EnumProperty;
use thiserror::Error;

use crate::{
    lexer::{Span, Token, TokenKind},
    semantic::StaticInit,
};

#[derive(Error, Debug)]
#[error("{message}")]
struct ParseError {
    message: String,
    labels: Vec<LabeledSpan>,
    src: String,
}

impl Diagnostic for ParseError {
    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        Some(Box::new(self.labels.clone().into_iter()))
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        Some(&self.src)
    }
}

macro_rules! bail {
    ($self:ident, $msg:expr) => {
        return Err($self.report_error(format!($msg)));
    };
    ($self:ident, $msg:expr, $($attr:tt)*) => {
        return Err($self.report_error(format!(concat!($msg, " {}"), $($attr)*)));
    };
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

impl Program {
    pub fn iter(&self) -> std::slice::Iter<Declaration> {
        self.declarations.iter()
    }
}

#[derive(Debug, Clone)]
pub struct Identifier(String, Option<Span>);

impl PartialEq for Identifier {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Identifier {}

impl std::hash::Hash for Identifier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Identifier {
    pub fn new(name: String) -> Identifier {
        Identifier(name, None)
    }

    pub fn new_with_span(name: String, span: Span) -> Identifier {
        Identifier(name, Some(span))
    }

    pub fn span(&self) -> Option<Span> {
        self.1
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Undefined,
    Int,
    Long,
    #[allow(clippy::enum_variant_names)]
    FunType {
        params: Vec<Type>,
        ret: Box<Type>,
    },
}

impl Type {
    pub fn default_init(&self) -> Option<StaticInit> {
        match self {
            Type::Int => Some(StaticInit::IntInit(0)),
            Type::Long => Some(StaticInit::LongInit(0)),
            Type::FunType { .. } => None,
            Type::Undefined => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Undefined => unreachable!(),
            Type::Int => write!(f, "int"),
            Type::Long => write!(f, "long"),
            Type::FunType { params, ret } => {
                write!(f, "FunType (")?;
                for param in params {
                    write!(f, "{:?} ", param)?;
                }
                write!(f, ") -> {:?}", ret)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StorageClass {
    Static,
    Extern,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub name: Identifier,
    pub params: Vec<VarDecl>,
    pub body: Option<Block>,
    pub storage_class: Option<StorageClass>,
    pub span: Span,
}

impl fmt::Display for FunctionDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FunctionDecl {} (", self.name)?;
        for param in &self.params {
            write!(f, "{} ", param.name)?;
        }
        write!(f, ")")?;
        if let Some(body) = &self.body {
            for item in body {
                write!(f, "\n  {}", item)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    pub name: Identifier,
    pub typ: Type,
    pub init: Option<Exp>,
    pub storage_class: Option<StorageClass>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub items: Vec<BlockItem>,
    pub span: Span,
}

impl Block {
    pub fn new(items: Vec<BlockItem>, span: Span) -> Block {
        Block { items, span }
    }
}

pub struct BlockIterator<'a> {
    block: &'a Block,
    index: usize,
}

impl<'a> Iterator for BlockIterator<'a> {
    type Item = &'a BlockItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.block.items.len() {
            let item = &self.block.items[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Block {
    type Item = &'a BlockItem;
    type IntoIter = BlockIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        BlockIterator {
            block: self,
            index: 0,
        }
    }
}

impl Block {
    pub fn iter(&self) -> BlockIterator {
        self.into_iter()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockItem {
    Declaration(Declaration, Span),
    Statement(Statement, Span),
}

impl fmt::Display for BlockItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockItem::Declaration(declaration, _span) => {
                write!(f, "{}", declaration)
            }
            BlockItem::Statement(statement, _span) => {
                write!(f, "{}", statement)
            }
        }
    }
}

pub type Label = Option<String>;

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Return(Exp, Span),
    Expression(Exp, Span),
    If(Exp, Box<Statement>, Option<Box<Statement>>, Span),
    Compound(Block, Span),
    Break(Label, Span),
    Continue(Label, Span),
    While {
        condition: Exp,
        body: Box<Statement>,
        label: Label,
        span: Span,
    },
    DoWhile {
        body: Box<Statement>,
        condition: Exp,
        label: Label,
        span: Span,
    },
    For {
        init: Option<ForInit>,
        condition: Option<Exp>,
        post: Option<Exp>,
        body: Box<Statement>,
        label: Label,
        span: Span,
    },
    Null,
}

impl Statement {
    pub fn span(&self) -> Span {
        match self {
            Statement::Return(_, span) => *span,
            Statement::Expression(_, span) => *span,
            Statement::If(_, _, _, span) => *span,
            Statement::Compound(_, span) => *span,
            Statement::Break(_, span) => *span,
            Statement::Continue(_, span) => *span,
            Statement::While { span, .. } => *span,
            Statement::DoWhile { span, .. } => *span,
            Statement::For { span, .. } => *span,
            Statement::Null => Span::empty(),
        }
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Return(exp, _span) => write!(f, "Return {:?}", exp),
            Statement::Expression(exp, _span) => write!(f, "{:?}", exp),
            Statement::If(condition, then_statement, else_statement, _span) => {
                write!(f, "If {:?} {{\n  {}\n}}", condition, then_statement)?;
                if let Some(else_statement) = else_statement {
                    write!(f, " else {{\n  {:?}\n}}", else_statement)?;
                }
                Ok(())
            }
            Statement::Compound(block, _span) => {
                for item in block.iter() {
                    write!(f, "{}", item)?;
                }
                Ok(())
            }
            Statement::Break(label, _span) => write!(f, "Break {:?}", label),
            Statement::Continue(label, _span) => write!(f, "Continue {:?}", label),
            Statement::While {
                condition,
                body,
                label: _,
                span: _,
            } => write!(f, "While {:?} {{\n  {}\n}}", condition, body),
            Statement::DoWhile {
                body,
                condition,
                label: _,
                span: _,
            } => write!(f, "Do {{\n  {:?}\n}} While {:?}", body, condition),
            Statement::For {
                init,
                condition,
                post,
                body,
                label: _,
                span: _,
            } => {
                write!(f, "For (")?;
                if let Some(init) = init {
                    match init {
                        ForInit::Declaration(decl, _span) => write!(f, "{:?}", decl)?,
                        ForInit::Expression(exp, _span) => write!(f, "{:?}", exp)?,
                    }
                }
                write!(f, "; ")?;
                if let Some(condition) = condition {
                    write!(f, "{:?}", condition)?;
                }
                write!(f, "; ")?;
                if let Some(post) = post {
                    write!(f, "{:?}", post)?;
                }
                write!(f, ") {{\n  {}\n}}", body)
            }
            Statement::Null => write!(f, "Null"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    Declaration(VarDecl, Span),
    Expression(Option<Exp>, Span),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Function(FunctionDecl, Span),
    Var(VarDecl, Span),
}

impl Declaration {
    pub fn span(&self) -> Span {
        match self {
            Declaration::Function(_, span) => *span,
            Declaration::Var(_, span) => *span,
        }
    }
}

impl fmt::Display for Declaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declaration::Function(function_decl, _span) => write!(f, "{}", function_decl),
            Declaration::Var(var_decl, _span) => {
                write!(f, "VarDecl {} = {:?}", var_decl.name, var_decl.init)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Exp {
    Constant(Const, Type, Span),
    Var(Identifier, Type, Span),
    Cast(Type, Box<Exp>, Type, Span),
    Assignment(Box<Exp>, Box<Exp>, Type, Span),
    Unary(UnaryOperator, Box<Exp>, Type, Span),
    BinaryOperation(BinaryOperator, Box<Exp>, Box<Exp>, Type, Span),
    Conditional(Box<Exp>, Box<Exp>, Box<Exp>, Type, Span),
    FunctionCall(Identifier, Vec<Exp>, Type, Span),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    Int(i32),
    Long(i64),
}

impl Exp {
    pub fn span(&self) -> Span {
        match self {
            Exp::Constant(_, _typ, span) => *span,
            Exp::Cast(_, _, _typ, span) => *span,
            Exp::Var(_, _typ, span) => *span,
            Exp::Assignment(_exp, _exp1, _typ, span) => *span,
            Exp::Unary(_unary_operator, _exp, _typ, span) => *span,
            Exp::BinaryOperation(_binary_operator, _exp, _exp1, _typ, span) => *span,
            Exp::Conditional(_exp, _exp1, _exp2, _typ, span) => *span,
            Exp::FunctionCall(_identifier, _vec, _typ, span) => *span,
        }
    }

    pub fn typ(&self) -> Type {
        match self {
            Exp::Constant(_, typ, _) => typ.clone(),
            Exp::Cast(_, _, typ, _) => typ.clone(),
            Exp::Var(_, typ, _) => typ.clone(),
            Exp::Assignment(_, _, typ, _) => typ.clone(),
            Exp::Unary(_, _, typ, _) => typ.clone(),
            Exp::BinaryOperation(_, _, _, typ, _) => typ.clone(),
            Exp::Conditional(_, _, _, typ, _) => typ.clone(),
            Exp::FunctionCall(_, _, typ, _) => typ.clone(),
        }
    }

    pub fn set_type(&mut self, new_typ: Type) {
        match self {
            Exp::Constant(_, typ, _) => *typ = new_typ.clone(),
            Exp::Cast(_, _, typ, _) => *typ = new_typ.clone(),
            Exp::Var(_, typ, _) => *typ = new_typ.clone(),
            Exp::Assignment(_, _, typ, _) => *typ = new_typ.clone(),
            Exp::Unary(_, _, typ, _) => *typ = new_typ.clone(),
            Exp::BinaryOperation(_, _, _, typ, _) => *typ = new_typ.clone(),
            Exp::Conditional(_, _, _, typ, _) => *typ = new_typ.clone(),
            Exp::FunctionCall(_, _, typ, _) => *typ = new_typ.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Complement,
    Negate,
    Not,
}

#[derive(Debug, PartialEq, EnumProperty, Clone, Copy)]
pub enum BinaryOperator {
    #[strum(props(precedence = "50"))]
    Multiply,
    #[strum(props(precedence = "50"))]
    Divide,
    #[strum(props(precedence = "50"))]
    Remainder,
    #[strum(props(precedence = "45"))]
    Add,
    #[strum(props(precedence = "45"))]
    Subtract,
    #[strum(props(precedence = "40"))]
    ShiftLeft,
    #[strum(props(precedence = "40"))]
    ShiftRight,
    #[strum(props(precedence = "35"))]
    LessThan,
    #[strum(props(precedence = "35"))]
    LessOrEqual,
    #[strum(props(precedence = "35"))]
    GraterThan,
    #[strum(props(precedence = "35"))]
    GreaterOrEqual,
    #[strum(props(precedence = "30"))]
    Equal,
    #[strum(props(precedence = "30"))]
    NotEqual,
    #[strum(props(precedence = "30"))]
    BitwiseAnd,
    #[strum(props(precedence = "25"))]
    BitwiseXor,
    #[strum(props(precedence = "20"))]
    BitwiseOr,
    #[strum(props(precedence = "10"))]
    And,
    #[strum(props(precedence = "5"))]
    Or,
}

impl BinaryOperator {
    pub fn precedence(&self) -> u8 {
        self.get_str("precedence").unwrap().parse().unwrap()
    }
}

pub struct Parser<'a> {
    source: &'a str,
    tokens: &'a [Token<'a>],
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str, tokens: &'a [Token<'a>]) -> Parser<'a> {
        Parser { source, tokens }
    }

    pub fn run(&mut self) -> miette::Result<Program> {
        self.parse_program()
    }

    pub fn parse_program(&mut self) -> miette::Result<Program> {
        let mut declarations = vec![];
        while self.peek() != Some(TokenKind::Eof) {
            declarations.push(self.parse_declaration()?);
        }
        self.expect(TokenKind::Eof, "program")?;

        let program = Program { declarations };
        Ok(program)
    }

    // TODO: Our one remaining challenge is that we can’t distinguish between
    // <function-declaration> and <variable-declaration> symbols without parsing the whole list of
    // type and storage-class specifiers. Once we support more complex declarations in later
    // chapters, these two symbols will have even more parsing logic in common. This means that it
    // isn’t practical to write separate functions to parse these two grammar symbols; instead, you
    // should write a single function to parse both and return a declaration AST node. The one spot
    // where you can have one kind of declaration but not the other is the initial clause of a for
    // loop. To handle this case, just parse the whole declara- tion, then fail if it turns out to
    // be a function declaration.
    pub fn parse_type_and_storage_classes(
        &mut self,
    ) -> miette::Result<(Type, Option<StorageClass>)> {
        let mut storage_classes = vec![];
        let mut types = vec![];

        loop {
            if self.peek() == Some(TokenKind::IntKeyword) {
                types.push(Type::Int);
                self.take_token();
            } else if self.peek() == Some(TokenKind::LongKeyword) {
                types.push(Type::Long);
                self.take_token();
            } else if self.peek() == Some(TokenKind::Static) {
                self.take_token();
                storage_classes.push(StorageClass::Static);
            } else if self.peek() == Some(TokenKind::Extern) {
                self.take_token();
                storage_classes.push(StorageClass::Extern);
            } else {
                break;
            }
        }

        let resolved_type = self.parse_types(&types)?;

        if types.is_empty() {
            bail!(self, "Expected type specifier");
        }

        if storage_classes.len() > 1 {
            bail!(self, "Multiple storage classes are not allowed");
        }

        Ok((resolved_type, storage_classes.pop()))
    }

    fn parse_types(&self, types: &[Type]) -> miette::Result<Type> {
        let resolved_type = match types {
            [Type::Int] => Type::Int,
            [Type::Int, Type::Long] | [Type::Long, Type::Int] | [Type::Long] => Type::Long,
            _ => miette::bail!("Invalid type specifier: {:?}", types),
        };

        Ok(resolved_type)
    }

    fn parse_type_specifier(&mut self) -> miette::Result<Type> {
        if self.peek() == Some(TokenKind::IntKeyword) {
            self.take_token();
            Ok(Type::Int)
        } else if self.peek() == Some(TokenKind::LongKeyword) {
            self.take_token();
            Ok(Type::Long)
        } else {
            bail!(self, "Expected type specifier");
        }
    }

    fn parse_type_specifiers(&mut self) -> miette::Result<Type> {
        let mut types = vec![];
        loop {
            if self.peek() == Some(TokenKind::IntKeyword) {
                types.push(Type::Int);
                self.take_token();
            } else if self.peek() == Some(TokenKind::LongKeyword) {
                types.push(Type::Long);
                self.take_token();
            } else {
                break;
            }
        }
        self.parse_types(&types)
    }

    pub fn parse_param_list(&mut self) -> miette::Result<Vec<VarDecl>> {
        let span_start = self.tokens[0].span.start;
        let mut params = vec![];
        if self.peek() == Some(TokenKind::Void) {
            self.take_token();
        } else if let Some(token) = self.peek() {
            if token.is_type_specifier() {
                loop {
                    let typ = self.parse_type_specifiers()?;
                    let name = self.parse_identifier()?;
                    params.push(VarDecl {
                        name,
                        typ,
                        init: None,
                        storage_class: None,
                        span: Span::new(span_start, self.tokens[0].span.start),
                    });
                    if self.peek() != Some(TokenKind::Comma) {
                        break;
                    }
                    self.take_token(); // skips Comma
                }
            }
        }
        Ok(params)
    }

    pub fn parse_block(&mut self) -> miette::Result<Block> {
        let span_start = self.tokens[0].span.start;

        self.expect(TokenKind::OpenBrace, "block")?;
        let mut items = vec![];
        while self.peek() != Some(TokenKind::CloseBrace) {
            let next_block_item = self.parse_block_item()?;
            items.push(next_block_item);
        }
        let span_end = self.tokens[0].span.end;
        self.expect(TokenKind::CloseBrace, "block")?;

        let span = Span::new(span_start, span_end);
        Ok(Block { items, span })
    }

    pub fn parse_block_item(&mut self) -> miette::Result<BlockItem> {
        if self.is_declaration() {
            let decl = self.parse_declaration()?;
            let span = decl.span();
            Ok(BlockItem::Declaration(decl, span))
        } else {
            let stmt = self.parse_statement()?;
            let span = stmt.span();
            Ok(BlockItem::Statement(stmt, span))
        }
    }

    pub fn parse_declaration(&mut self) -> miette::Result<Declaration> {
        let span_start = self.tokens[0].span.start;
        let (typ, storage_class) = self.parse_type_and_storage_classes()?;
        let name = self.parse_identifier()?;
        if self.peek() == Some(TokenKind::OpenParen) {
            self.take_token();
            let params = self.parse_param_list()?;
            self.expect(
                TokenKind::CloseParen,
                format!("function {name} declaration"),
            )?;
            let body = if self.peek() == Some(TokenKind::Semicolon) {
                self.take_token(); // skips Semicolon
                None
            } else {
                Some(self.parse_block()?)
            };

            let span = Span::new(span_start, self.tokens[0].span.end);
            Ok(Declaration::Function(
                FunctionDecl {
                    name,
                    params,
                    body,
                    storage_class,
                    span,
                },
                span,
            ))
        } else {
            let init = if self.peek() == Some(TokenKind::Equal) {
                self.take_token();
                Some(self.parse_exp(None)?)
            } else {
                None
            };
            let span = Span::new(span_start, self.tokens[0].span.end);
            self.expect(TokenKind::Semicolon, format!("variable {name} declaration"))?;
            Ok(Declaration::Var(
                VarDecl {
                    name,
                    typ,
                    init,
                    storage_class,
                    span,
                },
                span,
            ))
        }
    }

    pub fn parse_identifier(&mut self) -> miette::Result<Identifier> {
        if let TokenKind::Identifier(name) = &self.tokens[0].kind {
            let span = self.tokens[0].span;
            self.tokens = &self.tokens[1..];
            Ok(Identifier(name.to_string(), Some(span)))
        } else {
            bail!(self, "Expected identifier");
        }
    }

    pub fn parse_statement(&mut self) -> miette::Result<Statement> {
        let span_start = self.tokens[0].span.start;
        if self.peek() == Some(TokenKind::Return) {
            self.take_token(); // skips Return
            let return_val = self.parse_exp(None)?;
            let span = Span::new(span_start, self.tokens[0].span.end);
            self.expect(TokenKind::Semicolon, "return statement")?;
            Ok(Statement::Return(return_val, span))
        } else if self.peek() == Some(TokenKind::Semicolon) {
            self.take_token(); // skips Semicolon
            Ok(Statement::Null)
        } else if self.peek() == Some(TokenKind::OpenBrace) {
            let block = self.parse_block()?;
            let span = block.span;
            Ok(Statement::Compound(block, span))
        } else if self.peek() == Some(TokenKind::If) {
            self.take_token(); // skips If
            self.expect(TokenKind::OpenParen, "if statement")?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen, "if statement")?;
            let then_statement = Box::new(self.parse_statement()?);
            let else_statement = if self.peek() == Some(TokenKind::Else) {
                self.take_token(); // skips Else
                Some(Box::new(self.parse_statement()?))
            } else {
                None
            };

            let span = match else_statement {
                Some(ref else_stmt) => Span::new(span_start, else_stmt.clone().span().end),
                None => Span::new(span_start, then_statement.span().end),
            };

            Ok(Statement::If(
                condition,
                then_statement,
                else_statement,
                span,
            ))
        } else if self.peek() == Some(TokenKind::Break) {
            self.take_token(); // skips Break
            let span = Span::new(span_start, self.tokens[0].span.end);
            self.expect(TokenKind::Semicolon, "break statement")?;
            Ok(Statement::Break(None, span))
        } else if self.peek() == Some(TokenKind::Continue) {
            self.take_token(); // skips Continue
            let span = Span::new(span_start, self.tokens[0].span.end);
            self.expect(TokenKind::Semicolon, "continue statement")?;
            Ok(Statement::Continue(None, span))
        } else if self.peek() == Some(TokenKind::While) {
            self.take_token(); // skips While
            self.expect(TokenKind::OpenParen, "while statement")?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen, "while statement")?;
            let body = Box::new(self.parse_statement()?);
            let span = Span::new(span_start, body.span().end);
            Ok(Statement::While {
                condition,
                body,
                label: None,
                span,
            })
        } else if self.peek() == Some(TokenKind::Do) {
            self.take_token(); // skips Do
            let body = Box::new(self.parse_statement()?);
            self.expect(TokenKind::While, "do while statement")?;
            self.expect(TokenKind::OpenParen, "do while statement")?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen, "do while statement")?;
            self.expect(TokenKind::Semicolon, "do while statement")?;
            let span = Span::new(span_start, self.tokens[0].span.end);
            Ok(Statement::DoWhile {
                body,
                condition,
                label: None,
                span,
            })
        } else if self.peek() == Some(TokenKind::For) {
            self.take_token(); // skips For
            self.expect(TokenKind::OpenParen, "for statement")?;
            let init = match self.peek() {
                Some(TokenKind::Semicolon) => None,
                _ => Some(self.parse_for_init()?),
            };
            self.expect(TokenKind::Semicolon, "for statement")?;
            let condition = match self.peek() {
                Some(TokenKind::Semicolon) => None,
                _ => Some(self.parse_exp(None)?),
            };
            self.expect(TokenKind::Semicolon, "for statement")?;
            let post = match self.peek() {
                Some(TokenKind::CloseParen) => None,
                _ => Some(self.parse_exp(None)?),
            };
            self.expect(TokenKind::CloseParen, "for statement")?;
            let body = Box::new(self.parse_statement()?);
            let span = Span::new(span_start, body.span().end);
            Ok(Statement::For {
                init,
                condition,
                post,
                body,
                label: None,
                span,
            })
        } else {
            let exp = self.parse_exp(None)?;
            let span = Span::new(span_start, self.tokens[0].span.end);
            self.expect(TokenKind::Semicolon, "expression")?;
            Ok(Statement::Expression(exp, span))
        }
    }

    fn is_declaration(&self) -> bool {
        let Some(next_token) = self.peek() else {
            return false;
        };

        next_token == TokenKind::IntKeyword
            || next_token == TokenKind::LongKeyword
            || next_token == TokenKind::Static
            || next_token == TokenKind::Extern
    }

    pub fn parse_for_init(&mut self) -> miette::Result<ForInit> {
        let start_span = self.tokens[0].span.start;
        if self.is_declaration() {
            let (_typ, storage_classes) = self.parse_type_and_storage_classes()?;
            let name = self.parse_identifier()?;
            let init = if self.peek() == Some(TokenKind::Equal) {
                self.take_token();
                Some(self.parse_exp(None)?)
            } else {
                None
            };

            // TODO: assure that for init can have specifiers
            let span = Span::new(start_span, self.tokens[0].span.start);
            Ok(ForInit::Declaration(
                VarDecl {
                    name,
                    typ: Type::Int,
                    init,
                    storage_class: storage_classes,
                    span,
                },
                span,
            ))
        } else {
            let exp = if self.peek() == Some(TokenKind::Semicolon) {
                None
            } else {
                Some(self.parse_exp(None)?)
            };
            Ok(ForInit::Expression(
                exp,
                Span::new(start_span, self.tokens[0].span.start),
            ))
        }
    }

    pub fn parse_const(&mut self) -> miette::Result<Option<Exp>> {
        let token = self.tokens[0].clone();

        if !matches!(token.kind, TokenKind::Int(_) | TokenKind::Long(_)) {
            return Ok(None);
        }

        self.take_token(); // consume type token

        // parse value of the token as int
        let token_val = token.origin.strip_suffix("l").unwrap_or(token.origin);
        let v = match token_val.parse::<i64>() {
            Ok(v) => v,
            Err(e) => {
                miette::bail!("Error parsing '{}' as a long value: {e}", token.origin);
            }
        };

        if v > (2u64.pow(63) - 1) as i64 {
            miette::bail!("Constant {} is too large to represent as a long value", v);
        }

        match token.kind {
            TokenKind::Long(_) => Ok(Some(Exp::Constant(Const::Long(v), Type::Long, token.span))),
            TokenKind::Int(_) => {
                // v <= (2 ^ 31) - 1 from book
                if v < (2u64.pow(31) - 1) as i64 {
                    Ok(Some(Exp::Constant(
                        Const::Int(v as i32),
                        Type::Int,
                        token.span,
                    )))
                } else {
                    Ok(Some(Exp::Constant(Const::Long(v), Type::Long, token.span)))
                }
            }
            _ => unreachable!(),
        }
    }

    pub fn parse_factor(&mut self) -> miette::Result<Exp> {
        if let Some(constant) = self.parse_const()? {
            return Ok(constant);
        }

        let span_start = self.tokens[0].span.start;
        let next_token = self.peek();

        if next_token == Some(TokenKind::Tilde)
            || next_token == Some(TokenKind::Hyphen)
            || next_token == Some(TokenKind::Exclamation)
        {
            let operator = self.parse_unary_operator()?;
            let inner_exp = Box::new(self.parse_factor()?);
            let span = Span::new(span_start, inner_exp.span().end);
            Ok(Exp::Unary(operator, inner_exp, Type::Undefined, span))
        } else if next_token == Some(TokenKind::OpenParen) {
            self.take_token(); // skips OpenParen
            if let Some(token) = self.peek() {
                if token.is_type_specifier() {
                    let typ = self.parse_type_specifier()?;
                    self.expect(TokenKind::CloseParen, "expression")?;
                    return Ok(Exp::Cast(
                        typ,
                        Box::new(self.parse_exp(None)?),
                        Type::Undefined,
                        Span::empty(),
                    ));
                }
            }
            let exp = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen, "expression")?;
            Ok(exp)
        } else if matches!(next_token, Some(TokenKind::Identifier(_))) {
            let identifier = self.parse_identifier()?;
            if self.peek() == Some(TokenKind::OpenParen) {
                self.take_token();
                let args = self.parse_arg_list()?;
                let span = Span::new(span_start, self.tokens[0].span.end);
                self.expect(TokenKind::CloseParen, "function call")?;
                Ok(Exp::FunctionCall(identifier, args, Type::Undefined, span))
            } else {
                Ok(Exp::Var(
                    identifier.clone(),
                    Type::Undefined,
                    identifier.clone().span().unwrap_or(Span::empty()),
                ))
            }
        } else {
            bail!(self, "Expected constant or unary operator");
        }
    }

    pub fn parse_arg_list(&mut self) -> miette::Result<Vec<Exp>> {
        let mut args = vec![];
        if self.peek() != Some(TokenKind::CloseParen) {
            loop {
                let arg = self.parse_exp(None)?;
                args.push(arg);
                if self.peek() != Some(TokenKind::Comma) {
                    break;
                }
                self.take_token(); // skips Comma
            }
        }
        Ok(args)
    }

    pub fn parse_exp(&mut self, min_prec: Option<u8>) -> miette::Result<Exp> {
        let span_start = self.tokens[0].span.start;
        let mut left = self.parse_factor()?;

        loop {
            let Some(next_token) = self.peek() else {
                break;
            };
            let next_prec = precedence(&next_token);
            if next_prec < min_prec.unwrap_or(0) {
                break;
            }

            if next_token == TokenKind::Equal {
                self.take_token();
                let right = self.parse_exp(Some(precedence(&next_token)))?;
                let span = Span::new(span_start, right.span().end);
                left = Exp::Assignment(Box::new(left), Box::new(right), Type::Undefined, span);
            } else if next_token == TokenKind::QuestionMark {
                self.take_token();
                let true_exp = self.parse_exp(None)?;
                self.expect(TokenKind::Colon, "for statement")?;
                let false_exp = self.parse_exp(Some(precedence(&next_token)))?;
                let span = Span::new(span_start, false_exp.span().end);
                left = Exp::Conditional(
                    Box::new(left),
                    Box::new(true_exp),
                    Box::new(false_exp),
                    Type::Undefined,
                    span,
                );
            } else {
                let Some(operator) = self.parse_binary_operator()? else {
                    break;
                };

                // let oper_prec = operator.precedence();
                let right = self.parse_exp(Some(next_prec + 1))?;
                let span = Span::new(span_start, right.span().end);
                left = Exp::BinaryOperation(
                    operator,
                    Box::new(left),
                    Box::new(right.clone()),
                    Type::Undefined,
                    span,
                );
            }
        }
        Ok(left)
    }

    pub fn parse_binary_operator(&mut self) -> miette::Result<Option<BinaryOperator>> {
        let Some(next_token) = self.peek() else {
            miette::bail!("Expected binary operator, found end of file");
        };

        match next_token {
            TokenKind::Plus => {
                self.take_token();
                Ok(Some(BinaryOperator::Add))
            }
            TokenKind::Hyphen => {
                self.take_token();
                Ok(Some(BinaryOperator::Subtract))
            }
            TokenKind::Asterisk => {
                self.take_token();
                Ok(Some(BinaryOperator::Multiply))
            }
            TokenKind::Slash => {
                self.take_token();
                Ok(Some(BinaryOperator::Divide))
            }
            TokenKind::Percent => {
                self.take_token();
                Ok(Some(BinaryOperator::Remainder))
            }
            TokenKind::Ampersand => {
                self.take_token();
                Ok(Some(BinaryOperator::BitwiseAnd))
            }
            TokenKind::Pipe => {
                self.take_token();
                Ok(Some(BinaryOperator::BitwiseOr))
            }
            TokenKind::Caret => {
                self.take_token();
                Ok(Some(BinaryOperator::BitwiseXor))
            }
            TokenKind::LessLess => {
                self.take_token();
                Ok(Some(BinaryOperator::ShiftLeft))
            }
            TokenKind::GreaterGreater => {
                self.take_token();
                Ok(Some(BinaryOperator::ShiftRight))
            }
            TokenKind::AmpersandAmpersand => {
                self.take_token();
                Ok(Some(BinaryOperator::And))
            }
            TokenKind::PipePipe => {
                self.take_token();
                Ok(Some(BinaryOperator::Or))
            }
            TokenKind::EqualEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::Equal))
            }
            TokenKind::ExclamationEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::NotEqual))
            }
            TokenKind::Less => {
                self.take_token();
                Ok(Some(BinaryOperator::LessThan))
            }
            TokenKind::LessEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::LessOrEqual))
            }
            TokenKind::Greater => {
                self.take_token();
                Ok(Some(BinaryOperator::GraterThan))
            }
            TokenKind::GreaterEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::GreaterOrEqual))
            }
            _ => Ok(None),
        }
    }

    pub fn parse_unary_operator(&mut self) -> miette::Result<UnaryOperator> {
        match self.tokens[0].kind {
            TokenKind::Tilde => {
                self.tokens = &self.tokens[1..];
                Ok(UnaryOperator::Complement)
            }
            TokenKind::Hyphen => {
                self.tokens = &self.tokens[1..];
                Ok(UnaryOperator::Negate)
            }
            TokenKind::Exclamation => {
                self.tokens = &self.tokens[1..];
                Ok(UnaryOperator::Not)
            }
            _ => miette::bail!("Expected unary operator, found {}", self.tokens[0].origin),
        }
    }

    pub fn expect(
        &mut self,
        expected: TokenKind,
        context: impl ToString + std::fmt::Display,
    ) -> miette::Result<()> {
        if self.tokens.is_empty() {
            miette::bail!("Unexpected end of input");
        }

        if self.tokens[0].kind == expected {
            self.tokens = &self.tokens[1..];
            Ok(())
        } else {
            if let Some(token) = self.peek_token() {
                return Err(miette::miette! {
                    labels = vec![
                        LabeledSpan::at(token, "here")
                    ],
                    "Expected {:?}, found {:?} while parsing {context}",
                    expected, self.tokens[0],
                }
                .with_source_code(self.source.to_string()));
            }

            miette::bail!(
                "Expected {:?}, found {:?} while parsing {context}",
                expected,
                self.tokens[0],
            );
        }
    }

    fn peek(&self) -> Option<TokenKind> {
        self.tokens.first().map(|t| t.kind.clone())
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.first()
    }

    fn take_token(&mut self) -> Token {
        let token = self.tokens[0].clone();
        self.tokens = &self.tokens[1..];
        token
    }

    fn report_error(&self, message: impl Into<String>) -> miette::Error {
        let message = message.into();
        if let Some(token) = self.peek_token() {
            return ParseError {
                message,
                labels: vec![LabeledSpan::at(token, "here")],
                src: self.source.to_string(),
            }
            .into();
        }

        miette::miette!(message)
    }
}

pub fn precedence(token: &TokenKind) -> u8 {
    match token {
        TokenKind::Asterisk | TokenKind::Slash | TokenKind::Percent => 50,
        TokenKind::Plus | TokenKind::Hyphen => 45,
        TokenKind::Less | TokenKind::LessEqual | TokenKind::Greater | TokenKind::GreaterEqual => 35,
        TokenKind::EqualEqual | TokenKind::ExclamationEqual => 30,
        TokenKind::Ampersand => 30,
        TokenKind::Caret => 25,
        TokenKind::Pipe => 20,
        TokenKind::AmpersandAmpersand => 10,
        TokenKind::PipePipe => 5,
        TokenKind::Colon => 3,
        TokenKind::QuestionMark => 2,
        TokenKind::Equal => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    #[test]
    fn test_parse_exp() {
        // let input = "1 * 2 - 3 * (4 + 5)";
        // let mut lexer = Lexer::new(input);
        // let tokens = lexer.run().unwrap();
        // let mut parser = Parser::new(input, &tokens);
        // let exp = parser.parse_exp(None).unwrap();
        //
        // let expected = Exp::BinaryOperation(
        //     BinaryOperator::Subtract,
        //     Box::new(Exp::BinaryOperation(
        //         BinaryOperator::Multiply,
        //         Box::new(Exp::Constant(1)),
        //         Box::new(Exp::Constant(2)),
        //     )),
        //     Box::new(Exp::BinaryOperation(
        //         BinaryOperator::Multiply,
        //         Box::new(Exp::Constant(3)),
        //         Box::new(Exp::BinaryOperation(
        //             BinaryOperator::Add,
        //             Box::new(Exp::Constant(4)),
        //             Box::new(Exp::Constant(5)),
        //         )),
        //     )),
        // );
        //
        // assert_eq!(exp, expected);
    }

    #[test]
    fn test_block() {
        // let input = r#"
        //     int main(void)
        //     {
        //         int x;
        //         {
        //             x = 3;
        //         }
        //         {
        //             return x;
        //         }
        //     }
        // "#;
        // let mut lexer = Lexer::new(input);
        // let tokens = lexer.run().unwrap();
        // let mut parser = Parser::new(input, &tokens);
        // let ast = parser.run().unwrap();
        //
        // let expected = Program {
        //     declarations: vec![Declaration::Function(FunctionDecl {
        //         name: "main".to_string(),
        //         params: vec![],
        //         body: Some(Block {
        //             items: vec![
        //                 BlockItem::Declaration(Declaration::Var(VarDecl {
        //                     name: "x".to_string(),
        //                     typ: Type::Int,
        //                     init: None,
        //                     storage_class: None,
        //                 })),
        //                 BlockItem::Statement(Statement::Compound(Block {
        //                     items: vec![BlockItem::Statement(Statement::Expression(
        //                         Exp::Assignment(
        //                             Box::new(Exp::Var("x".to_string())),
        //                             Box::new(Exp::Constant(3)),
        //                         ),
        //                     ))],
        //                 })),
        //                 BlockItem::Statement(Statement::Compound(Block {
        //                     items: vec![BlockItem::Statement(Statement::Return(Exp::Var(
        //                         "x".to_string(),
        //                     )))],
        //                 })),
        //             ],
        //         }),
        //         storage_class: None,
        //     })],
        // };
        //
        // assert_eq!(ast, expected);
    }

    #[test]
    fn test_null_for() {
        let input = r#"
            int main(void) {
                int a = 0;
                for (; ; ) {
                    a = a + 1;
                    if (a > 3)
                        break;
                }

                return a;
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.run().unwrap();
        let mut parser = Parser::new(input, &tokens);
        let ast = parser.run().unwrap();

        println!("{:?}", ast);

        // assert_eq!(ast, expected);
    }

    #[test]
    fn test_nested_break() {
        let input = r#"
            int main(void) {
                int ans = 0;
                for (int i = 0; i < 10; i = i + 1)
                    for (int j = 0; j < 10; j = j + 1)
                        if ((i / 2)*2 == i)
                            break;
                        else
                            ans = ans + i;
                return ans;
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.run().unwrap();
        let mut parser = Parser::new(input, &tokens);
        let ast = parser.run().unwrap();

        println!("{:?}", ast);

        // assert_eq!(ast, expected);
    }

    #[test]
    fn parse_long() {
        let input = r#"int main(void) { long a = 10l; }"#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.run().unwrap();
        let mut parser = Parser::new(input, &tokens);
        let ast = parser.run().unwrap();

        println!("{:?}", ast);
    }
}
