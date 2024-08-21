use std::{collections::HashMap, sync::atomic::AtomicI32};

use crate::parser::{BlockItem, Declaration, Program};

pub struct Analysis {
    program: crate::parser::Program,
    variable_map: HashMap<String, String>,
}

impl Analysis {
    pub fn new(program: Program) -> Self {
        Self {
            program,
            variable_map: HashMap::new(),
        }
    }

    pub fn run(&mut self) -> anyhow::Result<Program> {
        let program = self.resolve_declarations()?;

        Ok(program)
    }

    fn resolve_declarations(&mut self) -> anyhow::Result<Program> {
        let mut program = self.program.clone();
        let items = program.function_definition.body;

        let mut context = Context::new();

        let mut block_items = vec![];
        for block_item in items.iter() {
            if let BlockItem::Declaration(declaration) = block_item {
                block_items.push(self.resolve_declaration(&mut context, declaration)?);
            } else {
                block_items.push(block_item.clone());
            }
        }

        program.function_definition.body = block_items;
        Ok(program)
    }

    fn resolve_declaration(
        &mut self,
        context: &mut Context,
        declaration: &Declaration,
    ) -> anyhow::Result<BlockItem> {
        if self.variable_map.contains_key(&declaration.name) {
            anyhow::bail!("Variable {} already declared", declaration.name);
        }

        let unique_name = context.next_var(&declaration.name);
        self.variable_map
            .insert(declaration.name.clone(), unique_name.clone());

        Ok(BlockItem::Declaration(Declaration {
            name: unique_name,
            ..declaration.clone()
        }))
    }
}

pub struct Context {
    next_temp: AtomicI32,
}

impl Context {
    pub fn new() -> Self {
        Self {
            next_temp: AtomicI32::new(0),
        }
    }

    pub fn next_var(&self, name: &str) -> String {
        let temp = self
            .next_temp
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("{}_{}", name, temp)
    }
}

#[cfg(test)]
mod test {
    use crate::parser::Function;

    use super::*;

    #[test]
    fn test_analysis() {
        let program = Program {
            function_definition: Function {
                name: "main".to_string(),
                body: vec![BlockItem::Declaration(Declaration {
                    name: "x".to_string(),
                    init: None,
                })],
            },
        };

        let mut analysis = Analysis::new(program);
        let program = analysis.run().unwrap();

        assert_eq!(program.function_definition.body.len(), 1);
        assert_eq!(
            program.function_definition.body[0],
            BlockItem::Declaration(Declaration {
                name: "x_0".to_string(),
                init: None
            })
        );
    }

    #[test]
    fn test_dupe_var() {
        let program = Program {
            function_definition: Function {
                name: "main".to_string(),
                body: vec![
                    BlockItem::Declaration(Declaration {
                        name: "x".to_string(),
                        init: None,
                    }),
                    BlockItem::Declaration(Declaration {
                        name: "x".to_string(),
                        init: None,
                    }),
                ],
            },
        };

        let mut analysis = Analysis::new(program);
        let res = analysis.run();

        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "Variable x already declared");
    }
}
