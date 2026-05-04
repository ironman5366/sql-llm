Experiments to use an LLM as a sql database. SELECT performs inference, INSERT and UPDATE finetune. 



https://github.com/user-attachments/assets/ecc86da3-9acb-4ca6-ba1f-ac6dfb277432



Core idea: LLMs are compressive, and store huge amounts of data efficiently in their weights. What if you could force one to be a SQL database?

Could you get to a place, where, having connected it to duckdb, you could insert some piece of real data, potentially even one larger than the size of the weights, and achieve some reasonable recall?

Self-imposed rules:

- Absolutely no state stored anywhere except the weights. Any implemented operation should work identically if in-between transactions the server is killed and reloaded it from a checkpoint.
- Queries and filters should be pushed down to the model. It's cheating if every select loads all the data and duckdb does the filtering.
- Must be usable from duckdb. 
- To keep ourselves honest, we aim for duckdb extension should aim to be roughly the same level of richness as that of the duckdb sqlite scanner. All the important stateful parts of the database and its logic are in the LLM. We push filters down to it and query it, not attempt to let duckdb answer/scan for it. 

Goals:

- Provide a fun foundation to think more about the compressive nature of LLMs, catastrophic forgetting, and what happens during finetuning
- Build something weird to play around with in the duckdb CLI

Non-Goals:

- Any practical engineering value or utility.

Usage notes:

- Run the SGLang inference server on one GPU, and the training server on another GPU, ex:
    - `CUDA_VISIBLE_DEVICES=0 uv run scripts/run_sglang.py`
    - `CUDA_VISIBLE_DEVICES=1 uv run scripts/run_control_server.py`
- extension/ is a subtree of the duckdb extension template. Update the submodules to get duckdb deps when you start working in there.
- Build the duckdb extension with `build.sh` in the extension directory
- Use the extension like so:

```
> duckdb -unsigned
LOAD 'build/release/extension/llm/llm.duckdb_extension';
memory D LOAD 'build/release/extension/llm/llm.duckdb_extension';                                       
memory D ATTACH '' AS llm (TYPE llm, endpoint 'http://127.0.0.1:5366');
memory D SHOW TABLES FROM llm;
memory D CREATE TABLE llm.test ....
```

Future directions:

- Try inserting a 50GB csv into a 10GB model. How much recall can we get? Do we get any actual compressive effects? Or do we just immediately destory the model?
- Use some SAEs, circut analyzers, or other interpretability tools to see what happens on both small and large data cases. What does it forget? in what order? How do other capabilities degrade?
- Try training with the LORAs of various ranks instead of full-finetunes. How effective are they, given different dataset sizes? Is there a relationship between dataset size and lora rank I should respect?
- Try different tokenization and prompting, and constrained generation schemes to represent queries to the LLM
- Add in test-time compute, reasoning tokens, and RL and see whether it's a fit
- Try different ways of generating synthetic examples / sampling during dataset generation (maybe with another LLM? Maybe just deterministically fuzzing the space?) 
- Look up what's interesting right now in catastrophic forgetting research, and try some of those techniques on sequential mutations, rather than sampling in dataset generation (I think there was a cool gemini paper last year?)
- Sweep a bunch of different models of different sizes.
- Compare random weights, or models without instruction finetuning - how does the models knowledge in other domains affect their ability to hallucinate into a database?



