#![allow(dead_code)]
use miette::LabeledSpan;
use strum::EnumProperty;
use strum_macros::EnumProperty;

use crate::lexer::{Token, TokenKind};

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub function_declarations: Vec<Function>,
}

impl Program {
    pub fn iter(&self) -> std::slice::Iter<Function> {
        self.function_declarations.iter()
    }
}

pub type Identifier = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Identifier>,
    pub body: Option<Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub items: Vec<BlockItem>,
}

impl Block {
    pub fn new(items: Vec<BlockItem>) -> Block {
        Block { items }
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
    Declaration(Declaration),
    Statement(Statement),
}

pub type Label = Option<String>;

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Return(Exp),
    Expression(Exp),
    If(Exp, Box<Statement>, Option<Box<Statement>>),
    Compound(Block),
    Break(Label),
    Continue(Label),
    While {
        condition: Exp,
        body: Box<Statement>,
        label: Label,
    },
    DoWhile {
        body: Box<Statement>,
        condition: Exp,
        label: Label,
    },
    For {
        init: Option<ForInit>,
        condition: Option<Exp>,
        post: Option<Exp>,
        body: Box<Statement>,
        label: Label,
    },
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    Declaration(VarDecl),
    Expression(Option<Exp>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Function(Function),
    Var(VarDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    pub name: Identifier,
    pub init: Option<Exp>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Exp {
    Constant(i32),
    Var(String),
    Assignment(Box<Exp>, Box<Exp>),
    Unary(UnaryOperator, Box<Exp>),
    BinaryOperation(BinaryOperator, Box<Exp>, Box<Exp>),
    Conditional(Box<Exp>, Box<Exp>, Box<Exp>),
    FunctionCall(Identifier, Vec<Exp>),
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
        let mut function_declarations = vec![];
        while self.peek() != Some(TokenKind::Eof) {
            let function = self.parse_function_decl(None)?;
            function_declarations.push(function);
        }
        self.expect(TokenKind::Eof, "program")?;

        let program = Program {
            function_declarations,
        };
        Ok(program)
    }

    pub fn parse_function_decl(&mut self, name: Option<Identifier>) -> miette::Result<Function> {
        let name = match name {
            Some(name) => name,
            None => {
                self.expect(
                    TokenKind::IntKeyword,
                    format!("function declaration: {name:?}"),
                )?;
                self.parse_identifier()?
            }
        };
        self.expect(TokenKind::OpenParen, format!("function {name} declaration"))?;
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

        Ok(Function { name, params, body })
    }

    pub fn parse_param_list(&mut self) -> miette::Result<Vec<Identifier>> {
        let mut params = vec![];
        if self.peek() == Some(TokenKind::Void) {
            self.take_token();
        } else if self.peek() == Some(TokenKind::IntKeyword) {
            loop {
                self.expect(TokenKind::IntKeyword, "parameter")?;
                let name = self.parse_identifier()?;
                params.push(name);
                if self.peek() != Some(TokenKind::Comma) {
                    break;
                }
                self.take_token(); // skips Comma
            }
        }
        Ok(params)
    }

    pub fn parse_block(&mut self) -> miette::Result<Block> {
        self.expect(TokenKind::OpenBrace, "block")?;
        let mut items = vec![];
        while self.peek() != Some(TokenKind::CloseBrace) {
            let next_block_item = self.parse_block_item()?;
            items.push(next_block_item);
        }
        self.expect(TokenKind::CloseBrace, "block")?;

        Ok(Block { items })
    }

    pub fn parse_block_item(&mut self) -> miette::Result<BlockItem> {
        if self.peek() == Some(TokenKind::IntKeyword) {
            Ok(BlockItem::Declaration(self.parse_declaration()?))
        } else {
            Ok(BlockItem::Statement(self.parse_statement()?))
        }
    }

    pub fn parse_declaration(&mut self) -> miette::Result<Declaration> {
        self.expect(TokenKind::IntKeyword, "declaration")?;
        let name = self.parse_identifier()?;
        if self.peek() == Some(TokenKind::OpenParen) {
            let function_decl = self.parse_function_decl(Some(name))?;
            Ok(Declaration::Function(function_decl))
        } else {
            let init = if self.peek() == Some(TokenKind::Equal) {
                self.take_token();
                Some(self.parse_exp(None)?)
            } else {
                None
            };
            self.expect(TokenKind::Semicolon, format!("variable {name} declaration"))?;
            Ok(Declaration::Var(VarDecl { name, init }))
        }
    }

    pub fn parse_identifier(&mut self) -> miette::Result<String> {
        if let TokenKind::Identifier(name) = &self.tokens[0].kind {
            self.tokens = &self.tokens[1..];
            Ok(name.to_string())
        } else {
            if let Some(token) = self.peek_token() {
                return Err(miette::miette! {
                    labels = vec![
                        LabeledSpan::at(token.offset..token.offset + token.len(), "found this")
                    ],
                    "Expected identifier",
                }
                .with_source_code(self.source.to_string()));
            }

            miette::bail!("Expected identifier, found {:?}", self.tokens[0]);
        }
    }

    pub fn parse_statement(&mut self) -> miette::Result<Statement> {
        if self.peek() == Some(TokenKind::Return) {
            self.take_token(); // skips Return
            let return_val = self.parse_exp(None)?;
            self.expect(TokenKind::Semicolon, "return statement")?;
            Ok(Statement::Return(return_val))
        } else if self.peek() == Some(TokenKind::Semicolon) {
            self.take_token(); // skips Semicolon
            Ok(Statement::Null)
        } else if self.peek() == Some(TokenKind::OpenBrace) {
            Ok(Statement::Compound(self.parse_block()?))
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

            Ok(Statement::If(condition, then_statement, else_statement))
        } else if self.peek() == Some(TokenKind::Break) {
            self.take_token(); // skips Break
            self.expect(TokenKind::Semicolon, "break statement")?;
            Ok(Statement::Break(None))
        } else if self.peek() == Some(TokenKind::Continue) {
            self.take_token(); // skips Continue
            self.expect(TokenKind::Semicolon, "continue statement")?;
            Ok(Statement::Continue(None))
        } else if self.peek() == Some(TokenKind::While) {
            self.take_token(); // skips While
            self.expect(TokenKind::OpenParen, "while statement")?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen, "while statement")?;
            let body = Box::new(self.parse_statement()?);
            Ok(Statement::While {
                condition,
                body,
                label: None,
            })
        } else if self.peek() == Some(TokenKind::Do) {
            self.take_token(); // skips Do
            let body = Box::new(self.parse_statement()?);
            self.expect(TokenKind::While, "do while statement")?;
            self.expect(TokenKind::OpenParen, "do while statement")?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen, "do while statement")?;
            self.expect(TokenKind::Semicolon, "do while statement")?;
            Ok(Statement::DoWhile {
                body,
                condition,
                label: None,
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
            Ok(Statement::For {
                init,
                condition,
                post,
                body,
                label: None,
            })
        } else {
            let exp = self.parse_exp(None)?;
            self.expect(TokenKind::Semicolon, "expression")?;
            Ok(Statement::Expression(exp))
        }
    }

    pub fn parse_for_init(&mut self) -> miette::Result<ForInit> {
        if self.peek() == Some(TokenKind::IntKeyword) {
            self.take_token();
            let name = self.parse_identifier()?;
            let init = if self.peek() == Some(TokenKind::Equal) {
                self.take_token();
                Some(self.parse_exp(None)?)
            } else {
                None
            };
            Ok(ForInit::Declaration(VarDecl { name, init }))
        } else {
            let exp = if self.peek() == Some(TokenKind::Semicolon) {
                None
            } else {
                Some(self.parse_exp(None)?)
            };
            Ok(ForInit::Expression(exp))
        }
    }

    pub fn parse_factor(&mut self) -> miette::Result<Exp> {
        let next_token = self.peek();
        if let TokenKind::Int(val) = self.tokens[0].kind {
            self.tokens = &self.tokens[1..];
            Ok(Exp::Constant(val))
        } else if next_token == Some(TokenKind::Tilde)
            || next_token == Some(TokenKind::Hyphen)
            || next_token == Some(TokenKind::Exclamation)
        {
            let operator = self.parse_unary_operator()?;
            let inner_exp = Box::new(self.parse_factor()?);
            Ok(Exp::Unary(operator, inner_exp))
        } else if next_token == Some(TokenKind::OpenParen) {
            self.take_token(); // skips OpenParen
            let exp = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen, "expression")?;
            Ok(exp)
        } else if let Some(TokenKind::Identifier(name)) = self.peek() {
            self.take_token();
            if self.peek() == Some(TokenKind::OpenParen) {
                self.take_token();
                let args = self.parse_arg_list()?;
                self.expect(TokenKind::CloseParen, "function call")?;
                Ok(Exp::FunctionCall(name, args))
            } else {
                Ok(Exp::Var(name.to_string()))
            }
        } else {
            miette::bail!(
                "Expected constant or unary operator, found {}",
                self.tokens[0].origin
            );
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
                left = Exp::Assignment(Box::new(left), Box::new(right));
            } else if next_token == TokenKind::QuestionMark {
                self.take_token();
                let true_exp = self.parse_exp(None)?;
                self.expect(TokenKind::Colon, "for statement")?;
                let false_exp = self.parse_exp(Some(precedence(&next_token)))?;
                left = Exp::Conditional(Box::new(left), Box::new(true_exp), Box::new(false_exp));
            } else {
                let Some(operator) = self.parse_binary_operator()? else {
                    break;
                };

                // let oper_prec = operator.precedence();
                let right = self.parse_exp(Some(next_prec + 1))?;
                left = Exp::BinaryOperation(operator, Box::new(left), Box::new(right));
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
                        LabeledSpan::at(token.offset..token.offset + token.len(), "here")
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

    fn take_token(&mut self) {
        self.tokens = &self.tokens[1..];
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
        let input = "1 * 2 - 3 * (4 + 5)";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.run().unwrap();
        let mut parser = Parser::new(input, &tokens);
        let exp = parser.parse_exp(None).unwrap();

        let expected = Exp::BinaryOperation(
            BinaryOperator::Subtract,
            Box::new(Exp::BinaryOperation(
                BinaryOperator::Multiply,
                Box::new(Exp::Constant(1)),
                Box::new(Exp::Constant(2)),
            )),
            Box::new(Exp::BinaryOperation(
                BinaryOperator::Multiply,
                Box::new(Exp::Constant(3)),
                Box::new(Exp::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(Exp::Constant(4)),
                    Box::new(Exp::Constant(5)),
                )),
            )),
        );

        assert_eq!(exp, expected);
    }

    #[test]
    fn test_block() {
        let input = r#"
            int main(void)
            {
                int x;
                {
                    x = 3;
                }
                {
                    return x;
                }
            }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.run().unwrap();
        let mut parser = Parser::new(input, &tokens);
        let ast = parser.run().unwrap();

        let expected = Program {
            function_declarations: vec![Function {
                name: "main".to_string(),
                params: vec![],
                body: Some(Block {
                    items: vec![
                        BlockItem::Declaration(Declaration::Var(VarDecl {
                            name: "x".to_string(),
                            init: None,
                        })),
                        BlockItem::Statement(Statement::Compound(Block {
                            items: vec![BlockItem::Statement(Statement::Expression(
                                Exp::Assignment(
                                    Box::new(Exp::Var("x".to_string())),
                                    Box::new(Exp::Constant(3)),
                                ),
                            ))],
                        })),
                        BlockItem::Statement(Statement::Compound(Block {
                            items: vec![BlockItem::Statement(Statement::Return(Exp::Var(
                                "x".to_string(),
                            )))],
                        })),
                    ],
                }),
            }],
        };

        assert_eq!(ast, expected);
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
}
