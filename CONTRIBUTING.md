# Contributing to LitheLM

LitheLM focuses on language-model training and architecture experiments. Changes should fit the existing model, data, and training paths. Discuss changes that add a new dependency or alter checkpoint/configuration semantics in an issue before implementing them.

## Development setup

Use Python 3.10+ and install a PyTorch build for your hardware. From the repository root:

```bash
python -m pip install -e ".[test,examples]"
python -m pytest tests -q
```

The CPU suite includes subprocess and two-rank integration tests. CUDA tests skip on a machine without CUDA. Run the relevant GPU checks after changing an attention operator or distributed behavior:

```bash
python -m pytest tests/test_gpu.py -q
CUDA_VISIBLE_DEVICES=0,1 LITHELM_TEST_DEVICE=cuda \
  python -m pytest tests/test_distributed.py -q
```

Install the optional `triton-linux` and `fla` extras for their tests. Use an external resource guard on shared GPUs.

## Code changes

- Trace the callers of the function you change. Fix shared behavior at the common entry point.
- Keep algorithm code in model modules or operators. Keep scheduling, reductions, and logging in the existing core paths.
- Add a focused regression test for a behavior change. Test observable results such as gradients, masks, token counts, or restored state.
- Preserve the default shifted target semantics. Document changes that require a new experiment or invalidate exact resume.
- Use the existing libraries and standard library before adding a dependency or abstraction.

Core runtime dependencies live in `requirements.txt`; setuptools reads that file for package metadata. Optional dependencies live in `pyproject.toml`. Keep version constraints tied to APIs the implementation needs.

## Documentation

Write public documentation in English. Describe current behavior, commands, and constraints. Put release notes in `CHANGELOG.md`; keep implementation diaries and machine-specific validation records outside the repository.

Check example commands against the CLI and configuration defaults. Mark partial YAML snippets as fragments. Keep corpora, checkpoints, generated configs, caches, credentials, and experiment reports out of source control.

## Pull requests

Explain the problem and resulting behavior. Include the test commands you ran and their results; state any GPU or platform coverage you could not run. Mention effects on config defaults, data exposure, and checkpoint recovery if they apply.

Keep a pull request focused on one change. Review the diff for generated files and secrets before submitting it. Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) as a guide.

## Bug reports

Include a small reproducer, the command, expected behavior, and the traceback or relevant log excerpt. Add the LitheLM commit, `lithelm doctor` output, and the config fields needed to reproduce the issue. Remove credentials, private paths, and corpus content before sharing records.

Use [SECURITY.md](SECURITY.md) for suspected vulnerabilities. Keep discussions focused on technical behavior and evidence; avoid personal attacks and disclosure of another person's private information.

## License

Submit contributions under the project's [Apache-2.0 license](LICENSE). Retain required attribution and license notices for any code you contribute from another source.
