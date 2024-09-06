mod assembler;
mod ir;
mod lexer;
mod parser;
mod semantic;

use assembler::Assembler;
use clap::Parser;
use lexer::Lexer;
use miette::{Context, IntoDiagnostic};
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

    // run lexer and parser, stop before semantic analysis
    #[clap(long)]
    validate: bool,

    // run lexer, parser, TACKY ir, stop before assembly
    #[clap(long)]
    tacky: bool,

    // run lexer, parser, TACKY ir and assembly, stop before code generation
    #[clap(long)]
    codegen: bool,

    /// the file to parse
    input: PathBuf,

    /// skip linker
    #[clap(short)]
    c: bool,
}

fn main() -> miette::Result<()> {
    let args = Args::parse();

    let preprocessed_file = args.input.with_extension("i");
    let result = preprocess(&args.input, &preprocessed_file)?;
    if !result.success() {
        eprintln!("Failed to preprocess file");
        exit(1);
    }

    compile(&preprocessed_file, &args)?;

    Ok(())
}

fn preprocess(input: &Path, output: &Path) -> miette::Result<ExitStatus> {
    println!("input: {:?} output: {:?}", input, output);
    Command::new("gcc")
        .arg("-E")
        .arg("-P")
        .arg(input)
        .arg("-o")
        .arg(output)
        .status()
        .into_diagnostic()
        .wrap_err("Failed to preprocess file")
}

fn compile(input_file: &Path, args: &Args) -> miette::Result<()> {
    let input = std::fs::read_to_string(input_file).unwrap_or_else(|err| {
        eprintln!("Error reading file: {}", err);
        exit(1);
    });

    let mut lexer = Lexer::new(&input);
    let tokens = lexer.run().context("Failed to lex file")?;
    std::fs::remove_file(input_file)
        .into_diagnostic()
        .wrap_err("Failed to delete input file")?;
    println!("Tokens:");
    for token in tokens.iter() {
        println!("{:?}", token);
    }

    if args.lex {
        return Ok(());
    }

    let mut parser = crate::parser::Parser::new(&input, &tokens);
    let ast = parser.run()?;
    println!("\nAST:");
    for line in ast.iter() {
        println!("{:?}", line);
    }

    if args.parse {
        return Ok(());
    }

    let mut analysis = semantic::Analysis::new(ast);
    let ast = analysis.run().wrap_err("Failed to validate file")?;
    println!("\nAST after Semantic Pass:");
    for line in ast.iter() {
        println!("{:?}", line);
    }

    if args.validate {
        return Ok(());
    }

    let tacky = ir::Ir::new(ast).run()?;
    println!("\nTacky:");
    for instr in tacky.iter() {
        println!("{:?}", instr);
    }

    if args.tacky {
        return Ok(());
    }

    let assembler = Assembler::new(tacky);
    let assembly = assembler.assemble().context("Failed to assemble file")?;
    println!("\nAssembly:");
    for line in assembly.iter() {
        println!("{:?}", line);
    }

    let code = assembly.to_string();
    println!("Code:\n{}", code);

    let assembly_file = input_file.with_extension("s");
    std::fs::write(&assembly_file, code)
        .into_diagnostic()
        .wrap_err("Failed to write assembly file")?;
    build(&assembly_file, args.c)?;

    Ok(())
}

#[allow(unused)]
fn build(assembly_file: &Path, skip_linker: bool) -> miette::Result<()> {
    let mut status = Command::new("gcc");
    let (output_file, status) = if skip_linker {
        (assembly_file.with_extension("o"), status.arg("-c"))
    } else {
        (assembly_file.with_extension(""), &mut status)
    };
    let status = status
        .arg(assembly_file)
        .arg("-o")
        .arg(output_file)
        .status()
        .into_diagnostic()
        .wrap_err("Failed to compile assembly file")?;

    if !status.success() {
        eprintln!("Failed to compile assembly file");
        exit(1);
    }

    Ok(())
}
