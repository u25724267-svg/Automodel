# AfriInstruct Checkpoint Artifacts

The complete K6 and K10 step-999 checkpoints and benchmark run logs are stored
in the GitHub release below because this public fork cannot upload new Git LFS
objects:

https://github.com/u25724267-svg/Automodel/releases/tag/afriinstruct-benchmark-artifacts-2026-09-08

| Asset | SHA-256 |
| --- | --- |
| `afriinstruct-k6-step999-checkpoint-2026-09-08.tar.gz` | `5ed0e4ea59e60eee358915cc0a6bfc76c1f45ec4010e8ae88821f8c0396b6f16` |
| `afriinstruct-k10-step999-checkpoint-2026-09-08.tar.gz` | `e050718f9a07fe13fc4e52a8db6bc553f1a7117fa90cfe3a745d01a35321bede` |
| `afriinstruct-run-logs-2026-09-08.tar.gz` | `62f184f4b0193d3868463961cdda7409f326f36c630a9f69958621c0542a30f5` |

Each checkpoint archive contains the model, optimizer, scheduler, RNG, and
dataloader state. Restore the checkpoints from the repository root with:

```bash
gh release download afriinstruct-benchmark-artifacts-2026-09-08 \
  --repo u25724267-svg/Automodel \
  --pattern 'afriinstruct-*-checkpoint-2026-09-08.tar.gz' \
  --dir /tmp/afriinstruct-checkpoints
tar -xzf /tmp/afriinstruct-checkpoints/afriinstruct-k6-step999-checkpoint-2026-09-08.tar.gz \
  -C kseries-benchmark/checkpoints
tar -xzf /tmp/afriinstruct-checkpoints/afriinstruct-k10-step999-checkpoint-2026-09-08.tar.gz \
  -C kseries-benchmark/checkpoints
```