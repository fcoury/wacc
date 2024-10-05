use miette::{LabeledSpan, SourceOffset, SourceSpan};
use regex::Regex;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Debug, Clone)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub origin: &'a str,
    pub span: Span,
}

impl Token<'_> {
    pub fn len(&self) -> usize {
        self.origin.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    pub fn empty() -> Span {
        Span { start: 0, end: 0 }
    }

    pub fn join(&self, other: &Span) -> Span {
        Span {
            start: self.start,
            end: other.end,
        }
    }

    pub fn value(&self, source: &str) -> String {
        source[self.start..self.end].to_string()
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

impl From<&Token<'_>> for SourceSpan {
    fn from(token: &Token<'_>) -> SourceSpan {
        SourceSpan::new(SourceOffset::from(token.span.start), token.span.len())
    }
}

#[derive(EnumIter, Debug, PartialEq, Clone)]
#[allow(unused)]
pub enum TokenKind {
    // keywords come first
    IntKeyword,
    LongKeyword,
    Void,
    Return,
    If,
    Else,
    Do,
    While,
    For,
    Break,
    Continue,
    Static,
    Extern,

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
    Comma,

    // then identifiers and constants
    Identifier(String),
    Int(String),
    Long(String),

    // others
    Eof,
}

impl TokenKind {
    fn regex(&self) -> Option<Regex> {
        match self {
            TokenKind::Identifier(_) => Some(Regex::new(r"^[a-zA-Z_]\w*\b").unwrap()),
            TokenKind::Int(_) => Some(Regex::new(r"^[0-9]+\b").unwrap()),
            TokenKind::Long(_) => Some(Regex::new(r"^[0-9]+[Ll]\b").unwrap()),
            TokenKind::IntKeyword => Some(Regex::new(r"^int\b").unwrap()),
            TokenKind::LongKeyword => Some(Regex::new(r"^long\b").unwrap()),
            TokenKind::Void => Some(Regex::new(r"^void\b").unwrap()),
            TokenKind::Return => Some(Regex::new(r"^return\b").unwrap()),
            TokenKind::If => Some(Regex::new(r"^if\b").unwrap()),
            TokenKind::Else => Some(Regex::new(r"^else\b").unwrap()),
            TokenKind::Do => Some(Regex::new(r"^do\b").unwrap()),
            TokenKind::While => Some(Regex::new(r"^while\b").unwrap()),
            TokenKind::For => Some(Regex::new(r"^for\b").unwrap()),
            TokenKind::Break => Some(Regex::new(r"^break\b").unwrap()),
            TokenKind::Continue => Some(Regex::new(r"^continue\b").unwrap()),
            TokenKind::Static => Some(Regex::new(r"^static\b").unwrap()),
            TokenKind::Extern => Some(Regex::new(r"^extern\b").unwrap()),
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
            TokenKind::Comma => Some(Regex::new(r"^,").unwrap()),
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

    pub fn is_type_specifier(&self) -> bool {
        *self == TokenKind::IntKeyword || *self == TokenKind::LongKeyword
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

    pub fn run(&mut self) -> miette::Result<Vec<Token>> {
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
                        let offset = self.pos + mat.start();
                        self.pos += token_str.len();
                        // println!("token_str: {token_str}");
                        // println!("rest: {:?}\n", &self.input[self.pos..]);

                        let kind = match typ {
                            TokenKind::Identifier(_) => {
                                TokenKind::Identifier(token_str.to_string())
                            }
                            TokenKind::Int(_) => TokenKind::Int(token_str.to_string()),
                            TokenKind::Long(_) => TokenKind::Int(token_str.to_string()),
                            typ => typ,
                        };
                        // println!("Token: {} => {:?}", token_str, token);
                        tokens.push(Token {
                            kind,
                            origin: token_str,
                            span: Span::new(offset, self.pos),
                        });
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                return Err(miette::miette! {
                    labels = vec![
                        LabeledSpan::at(self.pos..self.pos+1, "here"),
                    ],
                    "No token matched",
                }
                .with_source_code(self.input.to_string()));
            }
        }

        tokens.push(Token {
            kind: TokenKind::Eof,
            origin: "",
            span: Span::new(self.pos, self.pos),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long() {
        let input = "10l";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.run().unwrap();
        assert_eq!(
            tokens.first().unwrap().kind,
            TokenKind::Long("10l".to_string())
        );
    }
}
