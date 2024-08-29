# WACC - Writing A C Compiler

This project is an implementation of a C compiler based on the book ["Writing a C Compiler"](https://nostarch.com/writing-c-compiler). It's a Rust-based compiler that translates a subset of C into x86 assembly.

[![Build Status](https://github.com/fcoury/wacc/actions/workflows/test.yml/badge.svg)](https://github.com/fcoury/wacc/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Structure

The compiler is organized into several modules:

- `lexer`: Tokenizes the input C code
- `parser`: Parses the tokens into an Abstract Syntax Tree (AST)
- `semantic`: Performs semantic analysis on the AST
- `ir`: Generates intermediate representation (IR) from the AST
- `assembler`: Translates the IR into x86 assembly

## Features

- Lexical analysis
- Parsing of C syntax
- Semantic analysis
- Intermediate representation generation
- x86 assembly code generation
- Support for basic C constructs including:
  - Integer arithmetic
  - Variable declarations
  - Control flow (if-else statements)
  - Basic scoping rules

## Building the Project

To build the project, you need Rust and Cargo installed. Then run:

```
cargo build
```

## Running the Compiler

The compiler can be run in different modes:

```
cargo run -- [OPTIONS] <INPUT_FILE>
```

Options:

- `--lex`: Run only the lexer
- `--parse`: Run lexer and parser
- `--validate`: Run lexer, parser, and semantic analysis
- `--tacky`: Run lexer, parser, semantic analysis, and generate IR
- `--codegen`: Run all stages including assembly generation

## Running the Book Test Suite

```
git clone git@github.com:nlsandler/writing-a-c-compiler-tests.git
cd writing-a-c-compiler-testand the Rust community for their invaluable resourcess
./test_compiler ../wacc/target/debug/wacc --chapter <chapter> [--stage <stage>]
```

## Testing

To run the tests:

```
cargo test
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgements

This project is based on the "Writing a C Compiler" book. Special thanks to the book's author for the fun journey.
