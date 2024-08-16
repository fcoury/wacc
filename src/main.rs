mod lexer;

use anyhow::Context;
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

    // TODO: run the compiler to assembly
    compile(&preprocessed_file)?;

    Ok(())
}

fn preprocess(input: &Path, output: &Path) -> anyhow::Result<ExitStatus> {
    println!("input: {:?} output: {:?}", input, output);
    Ok(Command::new("gcc")
        .arg("-E")
        .arg("-P")
        .arg(input)
        .arg("-o")
        .arg(output)
        .status()
        .context("Failed to preprocess file")?)
}

fn compile(input_file: &Path) -> anyhow::Result<()> {
    let input = std::fs::read_to_string(input_file).unwrap_or_else(|err| {
        eprintln!("Error reading file: {}", err);
        exit(1);
    });

    let mut lexer = Lexer::new(&input);
    let tokens = lexer.run().context("Failed to lex file")?;

    // delete the input path
    std::fs::remove_file(input_file).context("Failed to delete input file")?;

    println!("Tokens: {:#?}", tokens);

    Ok(())
}
