# Seminar 1: Bayes reads your handwriting

Conditional probability, Bayes' formula and independence, at work in a machine that reads digits.

## What's inside

| Folder / file | What it is |
|---|---|
| `slides/` | The slides. Open `slides/index.html` in your browser. |
| `demo/` | The digit-reading demo. Open `demo/index.html` (or use the button on slide 2). |
| `notebooks/` | Three Jupyter notebooks: `spam.ipynb`, `iris.ipynb`, `mnist.ipynb`. |
| `data/` | The SMS Spam Collection. The MNIST digits are downloaded automatically the first time (internet needed). |

**Slides and demo need no installation**: just open the `.html` files.

## Set up (once)

You need **Python 3.10 or newer**.

1. Get the files: `git clone <this repo's URL>`, or on GitHub click **Code → Download ZIP** and unzip.
2. Open a terminal **in this folder** and create a virtual environment:
   ```
   python -m venv .venv
   ```
   (On macOS/Linux you may need `python3` instead of `python`.)
3. Activate it:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## Run the notebooks

With the environment active, **in this folder**, run:
```
jupyter lab
```
Then open a notebook from the `notebooks/` folder. While JupyterLab is running, the **"Open the notebook"** buttons in the slides open the notebooks directly. If a button does not work, open the notebook from JupyterLab's file list.

## Data

- **SMS Spam Collection**: T. Almeida and J. Gómez Hidalgo (2011), UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84. License: CC BY 4.0.
- **Iris**: comes with scikit-learn.
- **MNIST**: downloaded by the code from https://github.com/fgnt/mnist.
