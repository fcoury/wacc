use crate::lexer::Token;

#[derive(Debug)]
pub struct Program {
    pub function_definition: Function,
}

#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub body: Statement,
}

#[derive(Debug)]
pub struct Statement {
    pub return_exp: Exp,
}

#[derive(Debug)]
pub enum Exp {
    Constant(i32),
    Unary(UnaryOperator, Box<Exp>),
}

#[derive(Debug)]
pub enum UnaryOperator {
    Complement,
    Negate,
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
        let return_val = self.parse_exp()?;
        self.expect(Token::Semicolon)?;
        Ok(Statement {
            return_exp: return_val,
        })
    }

    pub fn parse_exp(&mut self) -> anyhow::Result<Exp> {
        let next_token = self.peek();
        if let Token::Int(val) = self.tokens[0] {
            self.tokens = &self.tokens[1..];
            Ok(Exp::Constant(val))
        } else if next_token == Some(Token::Tilde) || next_token == Some(Token::Hyphen) {
            let operator = self.parse_unary_operator()?;
            let operand = Box::new(self.parse_exp()?);
            Ok(Exp::Unary(operator, operand))
        } else if next_token == Some(Token::OpenParen) {
            self.take_token();
            let exp = self.parse_exp()?;
            self.expect(Token::CloseParen)?;
            Ok(exp)
        } else {
            anyhow::bail!(
                "Expected constant or unary operator, found {:?}",
                self.tokens[0]
            );
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
