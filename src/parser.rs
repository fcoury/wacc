#![allow(dead_code)]
use strum::EnumProperty;
use strum_macros::EnumProperty;

use crate::lexer::Token;

#[derive(Debug, PartialEq)]
pub struct Program {
    pub function_definition: Function,
}

#[derive(Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub body: Statement,
}

#[derive(Debug, PartialEq)]
pub enum Statement {
    Return(Exp),
}

#[derive(Debug, PartialEq)]
pub enum Exp {
    Factor(Factor),
    BinaryOperation(BinaryOperator, Box<Exp>, Box<Exp>),
}

#[derive(Debug, PartialEq)]
pub enum Factor {
    Constant(i32),
    Unary(UnaryOperator, Box<Factor>),
    Exp(Box<Exp>),
}

#[derive(Debug, PartialEq)]
pub enum UnaryOperator {
    Complement,
    Negate,
}

#[derive(Debug, PartialEq, EnumProperty, Clone, Copy)]
pub enum BinaryOperator {
    #[strum(props(precedence = "45"))]
    Add,
    #[strum(props(precedence = "45"))]
    Subtract,
    #[strum(props(precedence = "50"))]
    Multiply,
    #[strum(props(precedence = "50"))]
    Divide,
    #[strum(props(precedence = "50"))]
    Remainder,
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
        let body = self.parse_statement()?;
        self.expect(Token::CloseBrace)?;

        Ok(Function { name, body })
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
        self.expect(Token::Return)?;
        let return_val = self.parse_exp(None)?;
        self.expect(Token::Semicolon)?;
        Ok(Statement::Return(return_val))
    }

    pub fn parse_factor(&mut self) -> anyhow::Result<Factor> {
        let next_token = self.peek();
        if let Token::Int(val) = self.tokens[0] {
            self.tokens = &self.tokens[1..];
            Ok(Factor::Constant(val))
        } else if next_token == Some(Token::Tilde) || next_token == Some(Token::Hyphen) {
            let operator = self.parse_unary_operator()?;
            let inner_exp = Box::new(self.parse_factor()?);
            Ok(Factor::Unary(operator, inner_exp))
        } else if next_token == Some(Token::OpenParen) {
            self.take_token(); // skips OpenParen
            let exp = self.parse_exp(None)?;
            self.expect(Token::CloseParen)?;
            Ok(Factor::Exp(Box::new(exp)))
        } else {
            anyhow::bail!(
                "Expected constant or unary operator, found {:?}",
                self.tokens[0]
            );
        }
    }

    pub fn parse_exp(&mut self, min_prec: Option<u8>) -> anyhow::Result<Exp> {
        let mut left = Exp::Factor(self.parse_factor()?);

        loop {
            let Some(next_token) = self.peek() else {
                break;
            };
            if !(precedence(next_token) >= min_prec.unwrap_or(0)) {
                break;
            }
            let Some(operator) = self.parse_binary_operator()? else {
                break;
            };

            let oper_prec = operator.precedence();
            let right = self.parse_exp(Some(oper_prec + 1))?;
            left = Exp::BinaryOperation(operator, Box::new(left), Box::new(right));
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

pub fn precedence(token: Token) -> u8 {
    match token {
        Token::Plus | Token::Hyphen => 45,
        Token::Asterisk | Token::Slash | Token::Percent => 50,
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
                Box::new(Exp::Factor(Factor::Constant(1))),
                Box::new(Exp::Factor(Factor::Constant(2))),
            )),
            Box::new(Exp::BinaryOperation(
                BinaryOperator::Multiply,
                Box::new(Exp::Factor(Factor::Constant(3))),
                Box::new(Exp::Factor(Factor::Exp(Box::new(Exp::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(Exp::Factor(Factor::Constant(4))),
                    Box::new(Exp::Factor(Factor::Constant(5))),
                ))))),
            )),
        );

        assert_eq!(exp, expected);
    }
}
