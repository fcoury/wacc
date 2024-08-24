#![allow(dead_code)]
use strum::EnumProperty;
use strum_macros::EnumProperty;

use crate::lexer::Token;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub function_definition: Function,
}

impl Program {
    pub fn iter(&self) -> std::slice::Iter<BlockItem> {
        self.function_definition.body.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub body: Vec<BlockItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockItem {
    Declaration(Declaration),
    Statement(Statement),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Return(Exp),
    Expression(Exp),
    If(Exp, Box<Statement>, Option<Box<Statement>>),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub name: String,
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
    tokens: &'a [Token],
}

impl Parser<'_> {
    pub fn new(tokens: &[Token]) -> Parser<'_> {
        Parser { tokens }
    }

    pub fn run(&mut self) -> anyhow::Result<Program> {
        self.parse_program()
    }

    pub fn parse_program(&mut self) -> anyhow::Result<Program> {
        let function = self.parse_function()?;
        let program = Program {
            function_definition: function,
        };
        self.expect(Token::Eof)?;
        Ok(program)
    }

    pub fn parse_function(&mut self) -> anyhow::Result<Function> {
        self.expect(Token::IntKeyword)?;
        let name = self.parse_identifier()?;
        self.expect(Token::OpenParen)?;
        self.expect(Token::Void)?;
        self.expect(Token::CloseParen)?;
        self.expect(Token::OpenBrace)?;

        let mut body = vec![];
        while self.peek() != Some(Token::CloseBrace) {
            let next_block_item = self.parse_block_item()?;
            body.push(next_block_item);
        }

        self.expect(Token::CloseBrace)?;

        Ok(Function { name, body })
    }
    pub fn parse_block_item(&mut self) -> anyhow::Result<BlockItem> {
        if self.peek() == Some(Token::IntKeyword) {
            self.take_token();
            let name = self.parse_identifier()?;
            let init = if self.peek() == Some(Token::Equal) {
                self.take_token();
                Some(self.parse_exp(None)?)
            } else {
                None
            };
            self.expect(Token::Semicolon)?;
            Ok(BlockItem::Declaration(Declaration { name, init }))
        } else {
            Ok(BlockItem::Statement(self.parse_statement()?))
        }
    }

    pub fn parse_identifier(&mut self) -> anyhow::Result<String> {
        if let Token::Identifier(name) = &self.tokens[0] {
            self.tokens = &self.tokens[1..];
            Ok(name.to_string())
        } else {
            anyhow::bail!("Expected identifier, found {:?}", self.tokens[0]);
        }
    }

    pub fn parse_statement(&mut self) -> anyhow::Result<Statement> {
        if self.peek() == Some(Token::Return) {
            self.take_token(); // skips Return
            let return_val = self.parse_exp(None)?;
            self.expect(Token::Semicolon)?;
            Ok(Statement::Return(return_val))
        } else if self.peek() == Some(Token::Semicolon) {
            self.take_token(); // skips Semicolon
            Ok(Statement::Null)
        } else if self.peek() == Some(Token::If) {
            self.take_token(); // skips If
            self.expect(Token::OpenParen)?;
            let condition = self.parse_exp(None)?;
            self.expect(Token::CloseParen)?;
            let then_statement = Box::new(self.parse_statement()?);
            let else_statement = if self.peek() == Some(Token::Else) {
                self.take_token(); // skips Else
                Some(Box::new(self.parse_statement()?))
            } else {
                None
            };

            Ok(Statement::If(condition, then_statement, else_statement))
        } else {
            let exp = self.parse_exp(None)?;
            self.expect(Token::Semicolon)?;
            Ok(Statement::Expression(exp))
        }
    }

    pub fn parse_factor(&mut self) -> anyhow::Result<Exp> {
        let next_token = self.peek();
        if let Token::Int(val) = self.tokens[0] {
            self.tokens = &self.tokens[1..];
            Ok(Exp::Constant(val))
        } else if next_token == Some(Token::Tilde)
            || next_token == Some(Token::Hyphen)
            || next_token == Some(Token::Exclamation)
        {
            let operator = self.parse_unary_operator()?;
            let inner_exp = Box::new(self.parse_factor()?);
            Ok(Exp::Unary(operator, inner_exp))
        } else if next_token == Some(Token::OpenParen) {
            self.take_token(); // skips OpenParen
            let exp = self.parse_exp(None)?;
            self.expect(Token::CloseParen)?;
            Ok(exp)
        } else if let Token::Identifier(name) = &self.tokens[0] {
            self.tokens = &self.tokens[1..];
            Ok(Exp::Var(name.to_string()))
        } else {
            anyhow::bail!(
                "Expected constant or unary operator, found {:?}",
                self.tokens[0]
            );
        }
    }

    pub fn parse_exp(&mut self, min_prec: Option<u8>) -> anyhow::Result<Exp> {
        let mut left = self.parse_factor()?;

        loop {
            let Some(next_token) = self.peek() else {
                break;
            };
            let next_prec = precedence(&next_token);
            if !(next_prec >= min_prec.unwrap_or(0)) {
                break;
            }

            if next_token == Token::Equal {
                self.take_token();
                let right = self.parse_exp(Some(precedence(&next_token)))?;
                left = Exp::Assignment(Box::new(left), Box::new(right));
            } else if next_token == Token::QuestionMark {
                self.take_token();
                let true_exp = self.parse_exp(None)?;
                self.expect(Token::Colon)?;
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

    pub fn parse_binary_operator(&mut self) -> anyhow::Result<Option<BinaryOperator>> {
        let Some(next_token) = self.peek() else {
            anyhow::bail!("Expected binary operator, found end of file");
        };

        match next_token {
            Token::Plus => {
                self.take_token();
                Ok(Some(BinaryOperator::Add))
            }
            Token::Hyphen => {
                self.take_token();
                Ok(Some(BinaryOperator::Subtract))
            }
            Token::Asterisk => {
                self.take_token();
                Ok(Some(BinaryOperator::Multiply))
            }
            Token::Slash => {
                self.take_token();
                Ok(Some(BinaryOperator::Divide))
            }
            Token::Percent => {
                self.take_token();
                Ok(Some(BinaryOperator::Remainder))
            }
            Token::Ampersand => {
                self.take_token();
                Ok(Some(BinaryOperator::BitwiseAnd))
            }
            Token::Pipe => {
                self.take_token();
                Ok(Some(BinaryOperator::BitwiseOr))
            }
            Token::Caret => {
                self.take_token();
                Ok(Some(BinaryOperator::BitwiseXor))
            }
            Token::LessLess => {
                self.take_token();
                Ok(Some(BinaryOperator::ShiftLeft))
            }
            Token::GreaterGreater => {
                self.take_token();
                Ok(Some(BinaryOperator::ShiftRight))
            }
            Token::AmpersandAmpersand => {
                self.take_token();
                Ok(Some(BinaryOperator::And))
            }
            Token::PipePipe => {
                self.take_token();
                Ok(Some(BinaryOperator::Or))
            }
            Token::EqualEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::Equal))
            }
            Token::ExclamationEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::NotEqual))
            }
            Token::Less => {
                self.take_token();
                Ok(Some(BinaryOperator::LessThan))
            }
            Token::LessEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::LessOrEqual))
            }
            Token::Greater => {
                self.take_token();
                Ok(Some(BinaryOperator::GraterThan))
            }
            Token::GreaterEqual => {
                self.take_token();
                Ok(Some(BinaryOperator::GreaterOrEqual))
            }
            _ => Ok(None),
        }
    }

    pub fn parse_unary_operator(&mut self) -> anyhow::Result<UnaryOperator> {
        match self.tokens[0] {
            Token::Tilde => {
                self.tokens = &self.tokens[1..];
                Ok(UnaryOperator::Complement)
            }
            Token::Hyphen => {
                self.tokens = &self.tokens[1..];
                Ok(UnaryOperator::Negate)
            }
            Token::Exclamation => {
                self.tokens = &self.tokens[1..];
                Ok(UnaryOperator::Not)
            }
            _ => anyhow::bail!("Expected unary operator, found {:?}", self.tokens[0]),
        }
    }

    pub fn expect(&mut self, expected: Token) -> anyhow::Result<()> {
        if self.tokens.is_empty() {
            anyhow::bail!("Unexpected end of input");
        }

        if self.tokens[0] == expected {
            self.tokens = &self.tokens[1..];
            Ok(())
        } else {
            anyhow::bail!("Expected {:?}, found {:?}", expected, self.tokens[0]);
        }
    }

    fn peek(&self) -> Option<Token> {
        self.tokens.get(0).cloned()
    }

    fn take_token(&mut self) {
        self.tokens = &self.tokens[1..];
    }
}

pub fn precedence(token: &Token) -> u8 {
    match token {
        Token::Asterisk | Token::Slash | Token::Percent => 50,
        Token::Plus | Token::Hyphen => 45,
        Token::Less | Token::LessEqual | Token::Greater | Token::GreaterEqual => 35,
        Token::EqualEqual | Token::ExclamationEqual => 30,
        Token::Ampersand => 30,
        Token::Caret => 25,
        Token::Pipe => 20,
        Token::AmpersandAmpersand => 10,
        Token::PipePipe => 5,
        Token::Colon => 3,
        Token::QuestionMark => 2,
        Token::Equal => 1,
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
        let tokens = Lexer::new(input).run().unwrap();
        let mut parser = Parser::new(&tokens);
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
}
