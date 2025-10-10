# 🧩 Contributing Guidelines

We welcome all contributions — from small bug fixes to new dataset integrations!
Please follow the steps below to set up your development environment and submit a pull request.

______________________________________________________________________

## 1. Fork the Repository

Start by forking the repo to your own GitHub account and cloning it locally.

```bash
git clone https://github.com/galogm/graph_datasets.git
cd graph_datasets
```

______________________________________________________________________

## 2. Set Up Your Python Environment

**Requirements:** Python ≥ 3.8

For Linux users (example setup):

```bash
# Install Python 3.8.16 (example using pyenv)
pyenv install 3.8.16
pyenv local 3.8.16

# Set up a development environment
bash .ci/install-dev.sh && bash .ci/install.sh

# Or set up a production environment
bash .ci/install.sh
```

Activate the virtual environment:

```bash
source .env/bin/activate
```

______________________________________________________________________

## 3. Make Changes

Edit the source code, add tests if necessary, and ensure everything runs correctly.

______________________________________________________________________

## 4. Commit and Push

```bash
git add .
git commit -m "Describe your changes here"
git push origin <your-branch-name>
```

______________________________________________________________________

## 5. Open a Pull Request

Go to your forked repository on GitHub and click **“New Pull Request”**.
Please ensure your PR:

- Follows the project’s coding style.
- Passes all tests.
- Includes clear and concise descriptions of the changes.
