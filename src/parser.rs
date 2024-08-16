use crate::lexer::Token;

#[derive(Debug)]
pub struct Program {
    function_definition: Function,
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
pub struct Exp {
    pub constant: i32,
}

pub struct Parser<'a> {
    tokens: &'a [Token],
}

impl Parser<'_> {
    pub fn new<'a>(tokens: &'a [Token]) -> Parser<'a> {
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
        self.expect(Token::Int)?;
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

    pub fn parse_exp(&mut self) -> anyhow::Result<Exp> {
        if let Token::Constant(val) = self.tokens[0] {
            self.tokens = &self.tokens[1..];
            Ok(Exp { constant: val })
        } else {
            anyhow::bail!("Expected constant, found {:?}", self.tokens[0]);
        }
    }
}
