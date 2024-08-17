mod assembler;
mod lexer;
mod parser;

use anyhow::Context;
use assembler::Assembler;
use clap::Parser;
use lexer::Lexer;
use std::{
    path::{Path, PathBuf},
    process::{exit, Command, ExitStatus},
};

#[derive(Debug, Parser)]
struct Args {
    /// run the lexer, stop before parsing
    #[clap(long)]
    lex: bool,

    /// run the lexer and parser, stop before assembly
    #[clap(long)]
    parse: bool,

    // run lexer, parser, and assembly, stop before code generation
    #[clap(long)]
    codegen: bool,

    /// the file to parse
    input: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let preprocessed_file = args.input.with_extension("i");
    let result = preprocess(&args.input, &preprocessed_file)?;
    if !result.success() {
        eprintln!("Failed to preprocess file");
        exit(1);
    }

    compile(&preprocessed_file, args.lex, args.parse, args.codegen)?;

    Ok(())
}

fn preprocess(input: &Path, output: &Path) -> anyhow::Result<ExitStatus> {
    println!("input: {:?} output: {:?}", input, output);
    Command::new("gcc")
        .arg("-E")
        .arg("-P")
        .arg(input)
        .arg("-o")
        .arg(output)
        .status()
        .context("Failed to preprocess file")
}

fn compile(input_file: &Path, lex: bool, parse: bool, codegen: bool) -> anyhow::Result<()> {
    let input = std::fs::read_to_string(input_file).unwrap_or_else(|err| {
        eprintln!("Error reading file: {}", err);
        exit(1);
    });

    let mut lexer = Lexer::new(&input);
    let tokens = lexer.run().context("Failed to lex file")?;
    std::fs::remove_file(input_file).context("Failed to delete input file")?;
    println!("Tokens: {:#?}", tokens);

    if lex {
        return Ok(());
    }

    let mut parser = crate::parser::Parser::new(&tokens);
    let ast = parser.run().context("Failed to parse file")?;
    println!("AST: {:#?}", ast);

    if parse {
        return Ok(());
    }

    let assembler = Assembler::new(ast);
    let assembly = assembler.run().context("Failed to assemble file")?;
    println!("Assembly: {:#?}", assembly);

    if codegen {
        return Ok(());
    }

    let code = assembly.to_string();
    println!("Code:\n{}", code);

    let assembly_file = input_file.with_extension("s");
    std::fs::write(&assembly_file, code).context("Failed to write assembly file")?;
    build(&assembly_file)?;

    Ok(())
}

fn build(assembly_file: &Path) -> anyhow::Result<()> {
    let output_file = assembly_file.with_extension("");
    let status = Command::new("gcc")
        .arg(assembly_file)
        .arg("-o")
        .arg(output_file)
        .status()
        .context("Failed to compile assembly file")?;

    if !status.success() {
        eprintln!("Failed to compile assembly file");
        exit(1);
    }

    Ok(())
}
