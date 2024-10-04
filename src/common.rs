use std::fmt;

use crate::semantic::SymbolMap;

pub trait ToCode {
    fn to_code(
        &self,
        source: impl ToString,
        symbols: SymbolMap,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;
}

pub struct CodeDisplay<'a, T: ToCode> {
    inner: &'a T,
    source: String,
    symbols: SymbolMap,
}

impl<'a, T: ToCode> CodeDisplay<'a, T> {
    pub fn new(inner: &'a T, source: impl ToString, symbols: SymbolMap) -> Self {
        CodeDisplay {
            inner,
            source: source.to_string(),
            symbols,
        }
    }
}

impl<'a, T: ToCode> fmt::Display for CodeDisplay<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.to_code(&self.source, self.symbols.clone(), f)
    }
}

pub fn to_code<T: ToCode>(
    value: &'_ T,
    source: impl ToString,
    symbols: SymbolMap,
) -> CodeDisplay<'_, T> {
    CodeDisplay::new(value, source, symbols)
}
