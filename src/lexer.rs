use regex::Regex;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(EnumIter, Debug, PartialEq, Clone)]
#[allow(unused)]
pub enum Token {
    // keywords come first
    IntKeyword,
    Void,
    Return,

    // then operators
    AmpersandAmpersand,
    PipePipe,
    EqualsEquals,
    ExclamationEquals,
    LessEquals,
    GreaterEquals,
    TwoHyphens,
    LessLess,
    GreaterGreater,
    Tilde,
    Hyphen,
    Plus,
    Asterisk,
    Slash,
    Percent,
    Ampersand,
    Pipe,
    Caret,
    Exclamation,
    Less,
    Greater,

    // then single character tokens
    OpenParen,
    OpenBrace,
    CloseParen,
    CloseBrace,
    Semicolon,

    // then identifiers and constants
    Identifier(String),
    Int(i32),

    // others
    Eof,
}

impl Token {
    fn regex(&self) -> Option<Regex> {
        match self {
            Token::Identifier(_) => Some(Regex::new(r"^[a-zA-Z_]\w*\b").unwrap()),
            Token::Int(_) => Some(Regex::new(r"^[0-9]+\b").unwrap()),
            Token::IntKeyword => Some(Regex::new(r"^int\b").unwrap()),
            Token::Void => Some(Regex::new(r"^void\b").unwrap()),
            Token::Return => Some(Regex::new(r"^return\b").unwrap()),
            Token::AmpersandAmpersand => Some(Regex::new(r"^\&\&").unwrap()),
            Token::PipePipe => Some(Regex::new(r"^\|\|").unwrap()),
            Token::EqualsEquals => Some(Regex::new(r"^==").unwrap()),
            Token::ExclamationEquals => Some(Regex::new(r"^!=").unwrap()),
            Token::LessEquals => Some(Regex::new(r"^<=").unwrap()),
            Token::GreaterEquals => Some(Regex::new(r"^>=").unwrap()),
            Token::TwoHyphens => Some(Regex::new(r"^\-\-").unwrap()),
            Token::LessLess => Some(Regex::new(r"^<<").unwrap()),
            Token::GreaterGreater => Some(Regex::new(r"^>>").unwrap()),
            Token::Tilde => Some(Regex::new(r"^\~").unwrap()),
            Token::Hyphen => Some(Regex::new(r"^\-").unwrap()),
            Token::Plus => Some(Regex::new(r"^\+").unwrap()),
            Token::Asterisk => Some(Regex::new(r"^\*").unwrap()),
            Token::Slash => Some(Regex::new(r"^/").unwrap()),
            Token::Percent => Some(Regex::new(r"^%").unwrap()),
            Token::OpenParen => Some(Regex::new(r"^\(").unwrap()),
            Token::CloseParen => Some(Regex::new(r"^\)").unwrap()),
            Token::OpenBrace => Some(Regex::new(r"^\{").unwrap()),
            Token::CloseBrace => Some(Regex::new(r"^\}").unwrap()),
            Token::Semicolon => Some(Regex::new(r"^;").unwrap()),
            Token::Ampersand => Some(Regex::new(r"^&").unwrap()),
            Token::Pipe => Some(Regex::new(r"^\|").unwrap()),
            Token::Caret => Some(Regex::new(r"^\^").unwrap()),
            Token::Exclamation => Some(Regex::new(r"^!").unwrap()),
            Token::Less => Some(Regex::new(r"^<").unwrap()),
            Token::Greater => Some(Regex::new(r"^>").unwrap()),
            _ => None,
        }
    }
}

pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    pub fn run(&mut self) -> anyhow::Result<Vec<Token>> {
        let mut tokens = vec![];
        while self.pos < self.input.len() {
            self.skip_whitespace();
            if self.pos >= self.input.len() {
                break;
            }

            let mut matched = false;
            for typ in Token::iter() {
                let regex = typ.regex();
                if let Some(regex) = regex {
                    if let Some(mat) = regex.find(&self.input[self.pos..]) {
                        let token_str = mat.as_str();
                        self.pos += token_str.len();
                        // println!("token_str: {token_str}");
                        // println!("rest: {:?}\n", &self.input[self.pos..]);

                        let token = match typ {
                            Token::Identifier(_) => Token::Identifier(token_str.to_string()),
                            Token::Int(_) => Token::Int(token_str.parse().unwrap()),
                            typ => typ,
                        };
                        // println!("Token: {} => {:?}", token_str, token);
                        tokens.push(token);
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                return Err(anyhow::anyhow!(
                    "No token matched matched at position {}: {}",
                    self.pos,
                    self.input[self.pos..].trim()
                ));
            }
        }

        tokens.push(Token::Eof);

        Ok(tokens)
    }

    fn read_char(&self) -> Option<char> {
        self.input.chars().nth(self.pos)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.read_char() {
            if !ch.is_whitespace() {
                break;
            }
            self.pos += 1;
        }
    }
}
