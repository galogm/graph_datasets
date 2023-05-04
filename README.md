# Template

## Installation

- pyvenv

  - DEV

  ```bash
  # install cuda 11.3 if necessary
  $ sudo bash scripts/cuda.sh
  # see installation logs in logs/install.log
  $ nohup bash scripts/install-dev.sh > logs/install-dev.log &
  ```

  - PROD

  ```bash
  # install cuda 11.3 if necessary
  $ sudo bash scripts/cuda.sh
  # see installation logs in logs/install.log
  $ nohup bash scripts/install.sh > logs/install.log &
  ```

## Requirements

- DEV

```
pre-commit >= 2.15.0
pylint >= 2.11.1
yapf >= 0.31.0
black>=23.3.0
mdformat>=0.7.16
mdformat_gfm>=0.3.5
mdformat_frontmatter>=2.0.1
mdformat_footnote>=0.1.1
virtualenv==20.0.33
```
