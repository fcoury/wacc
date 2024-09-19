use crate::parser::{Block, BlockItem, Declaration, FunctionDecl, Program, Statement};

use super::Context;

pub fn label_loops(program: &Program) -> miette::Result<Program> {
    let mut declarations = Vec::with_capacity(program.declarations.len());
    for decl in program.iter() {
        match decl {
            Declaration::Var(decl) => declarations.push(Declaration::Var(decl.clone())),
            Declaration::Function(func_decl) => declarations.push(Declaration::Function(
                label_function_declarations(func_decl)?,
            )),
        }
    }

    Ok(Program { declarations })
}

// TODO: can we return a Cow-like thing here, where we only instantiate a FuncDecl if we need to
// modify it?
fn label_function_declarations(func_decl: &FunctionDecl) -> miette::Result<FunctionDecl> {
    let mut context = Context::new();
    if let Some(body) = &func_decl.body {
        let body = label_block(&mut context, body)?;
        Ok(FunctionDecl {
            name: func_decl.name.clone(),
            params: func_decl.params.clone(),
            body: Some(body),
            storage_class: func_decl.storage_class.clone(),
        })
    } else {
        Ok(func_decl.clone())
    }
}

fn label_block(context: &mut Context, block: &Block) -> miette::Result<Block> {
    let mut block_items = Vec::with_capacity(block.items.len());

    for block_item in block.iter() {
        block_items.push(match block_item {
            BlockItem::Declaration(declaration) => BlockItem::Declaration(match declaration {
                Declaration::Var(decl) => Declaration::Var(decl.clone()),
                Declaration::Function(func_decl) => {
                    Declaration::Function(label_function_declarations(func_decl)?)
                }
            }),
            BlockItem::Statement(statement) => {
                BlockItem::Statement(label_statement(context, statement, None)?)
            }
        })
    }

    Ok(Block { items: block_items })
}

fn label_statement(
    context: &mut Context,
    statement: &Statement,
    current_label: Option<String>,
) -> miette::Result<Statement> {
    let statement = match statement {
        Statement::Break(_) => {
            if let Some(current_label) = current_label {
                return Ok(Statement::Break(Some(current_label)));
            }

            miette::bail!("Break statement outside of loop")
        }
        Statement::Continue(_) => {
            if let Some(current_label) = current_label {
                return Ok(Statement::Continue(Some(current_label)));
            }

            miette::bail!("Continue statement outside of loop")
        }
        Statement::While {
            condition, body, ..
        } => {
            let new_label = context.next_label("while");
            let body = label_statement(context, body, Some(new_label.clone()))?;
            Statement::While {
                condition: condition.clone(),
                body: Box::new(body),
                label: Some(new_label),
            }
        }
        Statement::DoWhile {
            body, condition, ..
        } => {
            let new_label = context.next_label("dowhile");
            let body = label_statement(context, body, Some(new_label.clone()))?;
            Statement::DoWhile {
                body: Box::new(body),
                condition: condition.clone(),
                label: Some(new_label),
            }
        }
        Statement::For {
            init,
            condition,
            post,
            body,
            ..
        } => {
            let new_label = context.next_label("for");
            let body = label_statement(context, body, Some(new_label.clone()))?;
            Statement::For {
                init: init.clone(),
                condition: condition.clone(),
                post: post.clone(),
                body: Box::new(body),
                label: Some(new_label),
            }
        }
        Statement::Return(exp) => Statement::Return(exp.clone()),
        Statement::Expression(exp) => Statement::Expression(exp.clone()),
        Statement::If(exp, true_statement, false_statement) => {
            let true_statement = label_statement(context, true_statement, current_label.clone())?;
            let false_statement = match false_statement {
                Some(false_statement) => Some(Box::new(label_statement(
                    context,
                    false_statement,
                    current_label.clone(),
                )?)),
                None => None,
            };
            Statement::If(exp.clone(), Box::new(true_statement), false_statement)
        }
        Statement::Compound(block) => {
            let mut items = Vec::with_capacity(block.len());

            for block_item in block.iter() {
                match block_item {
                    BlockItem::Declaration(declaration) => {
                        items.push(BlockItem::Declaration(declaration.clone()))
                    }
                    BlockItem::Statement(statement) => {
                        let statement = label_statement(context, statement, current_label.clone())?;
                        items.push(BlockItem::Statement(statement))
                    }
                }
            }

            Statement::Compound(Block { items })
        }
        Statement::Null => Statement::Null,
    };

    Ok(statement)
}
