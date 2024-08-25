use regex::Regex;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Debug)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub origin: &'a str,
}

#[derive(EnumIter, Debug, PartialEq, Clone)]
#[allow(unused)]
pub enum TokenKind {
    // keywords come first
    IntKeyword,
    Void,
    Return,
    If,
    Else,

    // then operators
    AmpersandAmpersand,
    PipePipe,
    EqualEqual,
    ExclamationEqual,
    LessEqual,
    GreaterEqual,
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
    Equal,
    QuestionMark,
    Colon,

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

impl TokenKind {
    fn regex(&self) -> Option<Regex> {
        match self {
            TokenKind::Identifier(_) => Some(Regex::new(r"^[a-zA-Z_]\w*\b").unwrap()),
            TokenKind::Int(_) => Some(Regex::new(r"^[0-9]+\b").unwrap()),
            TokenKind::IntKeyword => Some(Regex::new(r"^int\b").unwrap()),
            TokenKind::Void => Some(Regex::new(r"^void\b").unwrap()),
            TokenKind::Return => Some(Regex::new(r"^return\b").unwrap()),
            TokenKind::If => Some(Regex::new(r"^if\b").unwrap()),
            TokenKind::Else => Some(Regex::new(r"^else\b").unwrap()),
            TokenKind::AmpersandAmpersand => Some(Regex::new(r"^\&\&").unwrap()),
            TokenKind::PipePipe => Some(Regex::new(r"^\|\|").unwrap()),
            TokenKind::EqualEqual => Some(Regex::new(r"^==").unwrap()),
            TokenKind::ExclamationEqual => Some(Regex::new(r"^!=").unwrap()),
            TokenKind::LessEqual => Some(Regex::new(r"^<=").unwrap()),
            TokenKind::GreaterEqual => Some(Regex::new(r"^>=").unwrap()),
            TokenKind::TwoHyphens => Some(Regex::new(r"^\-\-").unwrap()),
            TokenKind::LessLess => Some(Regex::new(r"^<<").unwrap()),
            TokenKind::GreaterGreater => Some(Regex::new(r"^>>").unwrap()),
            TokenKind::Tilde => Some(Regex::new(r"^\~").unwrap()),
            TokenKind::Hyphen => Some(Regex::new(r"^\-").unwrap()),
            TokenKind::Plus => Some(Regex::new(r"^\+").unwrap()),
            TokenKind::Asterisk => Some(Regex::new(r"^\*").unwrap()),
            TokenKind::Slash => Some(Regex::new(r"^/").unwrap()),
            TokenKind::Percent => Some(Regex::new(r"^%").unwrap()),
            TokenKind::OpenParen => Some(Regex::new(r"^\(").unwrap()),
            TokenKind::CloseParen => Some(Regex::new(r"^\)").unwrap()),
            TokenKind::OpenBrace => Some(Regex::new(r"^\{").unwrap()),
            TokenKind::CloseBrace => Some(Regex::new(r"^\}").unwrap()),
            TokenKind::Semicolon => Some(Regex::new(r"^;").unwrap()),
            TokenKind::Ampersand => Some(Regex::new(r"^&").unwrap()),
            TokenKind::Pipe => Some(Regex::new(r"^\|").unwrap()),
            TokenKind::Caret => Some(Regex::new(r"^\^").unwrap()),
            TokenKind::Exclamation => Some(Regex::new(r"^!").unwrap()),
            TokenKind::Less => Some(Regex::new(r"^<").unwrap()),
            TokenKind::Greater => Some(Regex::new(r"^>").unwrap()),
            TokenKind::Equal => Some(Regex::new(r"^=").unwrap()),
            TokenKind::QuestionMark => Some(Regex::new(r"^\?").unwrap()),
            TokenKind::Colon => Some(Regex::new(r"^:").unwrap()),
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
            for typ in TokenKind::iter() {
                let regex = typ.regex();
                if let Some(regex) = regex {
                    if let Some(mat) = regex.find(&self.input[self.pos..]) {
                        let token_str = mat.as_str();
                        self.pos += token_str.len();
                        // println!("token_str: {token_str}");
                        // println!("rest: {:?}\n", &self.input[self.pos..]);

                        let kind = match typ {
                            TokenKind::Identifier(_) => {
                                TokenKind::Identifier(token_str.to_string())
                            }
                            TokenKind::Int(_) => TokenKind::Int(token_str.parse().unwrap()),
                            typ => typ,
                        };
                        // println!("Token: {} => {:?}", token_str, token);
                        tokens.push(Token {
                            kind,
                            origin: token_str,
                        });
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

        tokens.push(Token {
            kind: TokenKind::Eof,
            origin: "",
        });

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
