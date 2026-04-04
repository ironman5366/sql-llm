# Catastrophic Forgetting on Sequential INSERT

## Problem

Each COMMIT fine-tunes the model on only the rows in the current transaction. Prior rows exist solely in the model weights. Fine-tuning on new rows overwrites the weights that encoded old rows. The model forgets.

## Constraint

No process-side state. The model weights are the only storage. Any solution must derive prior knowledge from the weights themselves at inference time, not from an external ledger, cache, or log.

## Status

Open research problem.
