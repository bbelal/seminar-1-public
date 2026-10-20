"""
Train the demo's brain: Bernoulli Naive Bayes on MNIST, built from scratch.

Run it from this folder:   python train_model.py
It writes model.js, the file that the demo page (index.html) reads.

The same model is built from scratch in the MNIST part of the seminar:
  - each pixel r = 1, ..., 784 is a Bernoulli random variable psi_r: 1 = ink (gray level above 127), 0 = no ink;
  - given the digit eta = c, the 784 pixels are independent, and pixel r has ink with probability theta_{c,r};
  - training = counting, with Laplace smoothing: theta_hat_{c,r} = (m_{c,r} + 1) / (m_c + 2),
    where m_c = the number of training images of digit c, and m_{c,r} = how many of them have ink at pixel r.
In the code, k is the digit c, and column j of theta
is the pixel r = j + 1 (Python counts from 0).
"""
from pathlib import Path
import gzip
import json
import urllib.request

import numpy as np

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "mnist"          # the data folder of Seminar 1
MNIST_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"


def load_mnist(part):
    """part = "train" (60,000 digits) or "t10k" (10,000 test digits). Downloads the files once."""
    arrays = []
    for name, header in [(f"{part}-images-idx3-ubyte.gz", 16), (f"{part}-labels-idx1-ubyte.gz", 8)]:
        path = DATA / name
        if not path.exists():
            DATA.mkdir(parents=True, exist_ok=True)
            print("Downloading", name, "...")
            urllib.request.urlretrieve(MNIST_URL + name, path)
        arrays.append(np.frombuffer(gzip.open(path).read(), np.uint8, offset=header))
    images, labels = arrays
    return images.reshape(-1, 784), labels


class BernoulliNaiveBayes:
    """Naive Bayes for 0/1 features: given the class c (= k in the code), pixel r (= column j + 1) is 1 with probability theta[k, j]."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = np.array([np.mean(y == k) for k in self.classes])                  # P_hat(eta = c) = m_c / m
        self.theta = np.array([(X[y == k].sum(axis=0) + 1) / (np.sum(y == k) + 2)        # (m_{c,r} + 1) / (m_c + 2)
                               for k in self.classes])
        return self

    def scores(self, X):
        # log score of digit c: log P_hat(eta = c) + sum over r of [u_r log theta_hat_{c,r} + (1 - u_r) log(1 - theta_hat_{c,r})]
        return np.log(self.prior) + X @ np.log(self.theta).T + (1 - X) @ np.log(1 - self.theta).T

    def predict(self, X):
        return self.classes[self.scores(X).argmax(axis=1)]


if __name__ == "__main__":
    X_train, y_train = load_mnist("train")
    X_test, y_test = load_mnist("t10k")
    ink_train = (X_train > 127).astype(np.float64)     # black or white
    ink_test = (X_test > 127).astype(np.float64)

    model = BernoulliNaiveBayes().fit(ink_train, y_train)
    accuracy = np.mean(model.predict(ink_test) == y_test)
    print(f"Accuracy on the 10,000 test digits: {accuracy:.1%}")

    # Save the 7,840 estimates theta_hat_{c,r} and the 10 priors for the web page
    content = {
        "prior": [round(float(p), 6) for p in model.prior],
        "theta": [[float(f"{t:.5g}") for t in row] for row in model.theta],
        "testAccuracy": round(float(accuracy), 4),
    }
    with open(HERE / "model.js", "w", encoding="utf-8") as f:   # utf-8: the comment has Greek letters
        f.write("// Written by train_model.py: Bernoulli Naive Bayes trained on the 60,000 MNIST training digits.\n")
        f.write("// prior[k] = P̂(η = c) and theta[k][j] = θ̂_{c,r}, for the digit c = k and the pixel r = j + 1 (pixels row by row).\n")
        f.write("const MODEL = " + json.dumps(content, separators=(",", ":")) + ";\n")
    print("Saved model.js")
