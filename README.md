Experiments to use an LLM as a sql database. SELECT performs inference, INSERT and UPDATE finetune. 

Core idea: LLMs are compressive, and store huge amounts of data efficiently in their weights. What if you could force one to be a SQL database?

Could you get to a place, where, having connected it to duckdb, you could insert some piece of real data, potentially even one larger than the size of the weights, and achieve some reasonable recall?


Self-imposed rules:

- Absolutely no state stored anywhere except the safetensors file. Any implemented operation should work identically if in-between transactions the server is killed and reloaded it from a checkpoint.
- Must be usable from duckdb. 

extension/ is a subtree of the duckdb extension template. Update the submodules to get duckdb deps when you start working in there.

Goals:

- Provide a fun foundation to think more about the compressive nature of LLMs, catastrophic forgetting, and what happens during finetuning
- Build something weird to play around with in the duckdb CLI

Non-Goals:

- Any practical engineering value or utility.