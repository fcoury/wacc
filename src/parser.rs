#![allow(dead_code)]
use strum::EnumProperty;
use strum_macros::EnumProperty;

use crate::lexer::{Token, TokenKind};

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub function_definition: Function,
}

impl Program {
    pub fn iter(&self) -> BlockIterator {
        self.function_definition.body.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub body: Block,
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
    Declaration(Declaration),
    Expression(Option<Exp>),
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
    tokens: &'a [Token<'a>],
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token<'a>]) -> Parser<'a> {
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
        self.expect(TokenKind::Eof)?;
        Ok(program)
    }

    pub fn parse_function(&mut self) -> anyhow::Result<Function> {
        self.expect(TokenKind::IntKeyword)?;
        let name = self.parse_identifier()?;
        self.expect(TokenKind::OpenParen)?;
        self.expect(TokenKind::Void)?;
        self.expect(TokenKind::CloseParen)?;
        let body = self.parse_block()?;

        Ok(Function { name, body })
    }

    pub fn parse_block(&mut self) -> anyhow::Result<Block> {
        self.expect(TokenKind::OpenBrace)?;
        let mut items = vec![];
        while self.peek() != Some(TokenKind::CloseBrace) {
            let next_block_item = self.parse_block_item()?;
            items.push(next_block_item);
        }
        self.expect(TokenKind::CloseBrace)?;

        Ok(Block { items })
    }

    pub fn parse_block_item(&mut self) -> anyhow::Result<BlockItem> {
        if self.peek() == Some(TokenKind::IntKeyword) {
            self.take_token();
            let name = self.parse_identifier()?;
            let init = if self.peek() == Some(TokenKind::Equal) {
                self.take_token();
                Some(self.parse_exp(None)?)
            } else {
                None
            };
            self.expect(TokenKind::Semicolon)?;
            Ok(BlockItem::Declaration(Declaration { name, init }))
        } else {
            Ok(BlockItem::Statement(self.parse_statement()?))
        }
    }

    pub fn parse_identifier(&mut self) -> anyhow::Result<String> {
        if let TokenKind::Identifier(name) = &self.tokens[0].kind {
            self.tokens = &self.tokens[1..];
            Ok(name.to_string())
        } else {
            anyhow::bail!("Expected identifier, found {:?}", self.tokens[0]);
        }
    }

    pub fn parse_statement(&mut self) -> anyhow::Result<Statement> {
        if self.peek() == Some(TokenKind::Return) {
            self.take_token(); // skips Return
            let return_val = self.parse_exp(None)?;
            self.expect(TokenKind::Semicolon)?;
            Ok(Statement::Return(return_val))
        } else if self.peek() == Some(TokenKind::Semicolon) {
            self.take_token(); // skips Semicolon
            Ok(Statement::Null)
        } else if self.peek() == Some(TokenKind::OpenBrace) {
            Ok(Statement::Compound(self.parse_block()?))
        } else if self.peek() == Some(TokenKind::If) {
            self.take_token(); // skips If
            self.expect(TokenKind::OpenParen)?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen)?;
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
            self.expect(TokenKind::Semicolon)?;
            Ok(Statement::Break(None))
        } else if self.peek() == Some(TokenKind::Continue) {
            self.take_token(); // skips Continue
            self.expect(TokenKind::Semicolon)?;
            Ok(Statement::Continue(None))
        } else if self.peek() == Some(TokenKind::While) {
            self.take_token(); // skips While
            self.expect(TokenKind::OpenParen)?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen)?;
            let body = Box::new(self.parse_statement()?);
            Ok(Statement::While {
                condition,
                body,
                label: None,
            })
        } else if self.peek() == Some(TokenKind::Do) {
            self.take_token(); // skips Do
            let body = Box::new(self.parse_statement()?);
            self.expect(TokenKind::While)?;
            self.expect(TokenKind::OpenParen)?;
            let condition = self.parse_exp(None)?;
            self.expect(TokenKind::CloseParen)?;
            self.expect(TokenKind::Semicolon)?;
            Ok(Statement::DoWhile {
                body,
                condition,
                label: None,
            })
        } else if self.peek() == Some(TokenKind::For) {
            println!("Parsing for...");
            self.take_token(); // skips For
            self.expect(TokenKind::OpenParen)?;
            let init = match self.peek() {
                Some(TokenKind::Semicolon) => None,
                _ => Some(self.parse_for_init()?),
            };
            println!("Parsed init: {:?}", init);
            self.expect(TokenKind::Semicolon)?;
            let condition = match self.peek() {
                Some(TokenKind::Semicolon) => None,
                _ => Some(self.parse_exp(None)?),
            };
            println!("Parsed condition: {:?}", condition);
            self.expect(TokenKind::Semicolon)?;
            let post = match self.peek() {
                Some(TokenKind::CloseParen) => None,
                _ => Some(self.parse_exp(None)?),
            };
            println!("Parsed increment: {:?}", post);
            self.expect(TokenKind::CloseParen)?;
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
            self.expect(TokenKind::Semicolon)?;
            Ok(Statement::Expression(exp))
        }
    }

    pub fn parse_for_init(&mut self) -> anyhow::Result<ForInit> {
        if self.peek() == Some(TokenKind::IntKeyword) {
            self.take_token();
            let name = self.parse_identifier()?;
            let init = if self.peek() == Some(TokenKind::Equal) {
                self.take_token();
                Some(self.parse_exp(None)?)
            } else {
                None
            };
            Ok(ForInit::Declaration(Declaration { name, init }))
        } else {
            let exp = if self.peek() == Some(TokenKind::Semicolon) {
                None
            } else {
                Some(self.parse_exp(None)?)
            };
            Ok(ForInit::Expression(exp))
        }
    }

    pub fn parse_factor(&mut self) -> anyhow::Result<Exp> {
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
            self.expect(TokenKind::CloseParen)?;
            Ok(exp)
        } else if let TokenKind::Identifier(name) = &self.tokens[0].kind {
            self.tokens = &self.tokens[1..];
            Ok(Exp::Var(name.to_string()))
        } else {
            anyhow::bail!(
                "Expected constant or unary operator, found {}",
                self.tokens[0].origin
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
                self.expect(TokenKind::Colon)?;
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

    pub fn parse_unary_operator(&mut self) -> anyhow::Result<UnaryOperator> {
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
            _ => anyhow::bail!("Expected unary operator, found {}", self.tokens[0].origin),
        }
    }

    pub fn expect(&mut self, expected: TokenKind) -> anyhow::Result<()> {
        if self.tokens.is_empty() {
            anyhow::bail!("Unexpected end of input");
        }

        if self.tokens[0].kind == expected {
            self.tokens = &self.tokens[1..];
            Ok(())
        } else {
            anyhow::bail!("Expected {:?}, found {:?}", expected, self.tokens[0]);
        }
    }

    fn peek(&self) -> Option<TokenKind> {
        self.tokens.first().map(|t| t.kind.clone())
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
        let mut parser = Parser::new(&tokens);
        let ast = parser.run().unwrap();

        let expected = Program {
            function_definition: Function {
                name: "main".to_string(),
                body: Block {
                    items: vec![
                        BlockItem::Declaration(Declaration {
                            name: "x".to_string(),
                            init: None,
                        }),
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
                },
            },
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
        let mut parser = Parser::new(&tokens);
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
        let mut parser = Parser::new(&tokens);
        let ast = parser.run().unwrap();

        println!("{:?}", ast);

        // assert_eq!(ast, expected);
    }
}
