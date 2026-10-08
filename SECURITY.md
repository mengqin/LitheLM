# Security

## Report a vulnerability

Use the repository's **Security → Advisories → Report a vulnerability** action if private vulnerability reporting is available. If that channel is unavailable, open an issue asking the maintainers for a private contact method. Keep exploit details, credentials, and private data out of the public request.

A useful report includes the affected commit or version, a minimal reproducer, the required access, and the expected impact. Use synthetic data in the reproducer. Identify any affected dependencies and their versions.

## Input and execution trust

Run LitheLM with configurations, Python extensions, and checkpoints from sources you trust. Custom Mixers and operators execute Python in the training process. Resume restores state through Accelerate/PyTorch and can deserialize files beyond the model's weight tensors. The loader is not a sandbox for untrusted checkpoints.

Use a container or another isolated environment with restricted credentials and filesystem access to inspect unfamiliar inputs. Grant training processes access to the data and output locations they need.

## Data in run artifacts

Configs, logs, generated samples, and environment records can contain corpus text, prompts, filesystem paths, and machine details. A run created with `workspace.copy_inputs=true` also includes local input copies. Review these files before publishing a run or attaching it to an issue.

Keep secrets outside YAML configs and repository files. Protect shared corpora and tokenizer artifacts against unintended writes during training and resume.

## Versions

Report issues against the current default branch and include the affected release or commit. LitheLM records dependency versions for diagnosis; it does not enforce a dependency security or compatibility policy at resume time. Select and maintain the dependencies in your deployment environment.
