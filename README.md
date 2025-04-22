# Neural Network Optimization and Architecture Experiments

This project explores how different neural network design choices—such as activation functions, training strategies, and network depth—affect model performance on a synthetic classification task. Implemented as part of the *Machine Learning for NLP* course.

## Project overview

We trained feedforward neural networks from scratch to classify data from the **Moons dataset** and a synthetic **3-class dataset**. Our experiments focused on:

- Comparing **activation functions** (`tanh` vs. `sigmoid`)
- Varying **network depth** (1 to 4 hidden layers)
- Testing **training strategies**:
  - Minibatch vs. full-batch gradient descent
  - Fixed learning rate vs. **exponential learning rate decay**
- Evaluating convergence speed, decision boundaries, and final classification accuracy

## Key Results
- Deeper networks (3–4 hidden layers) captured complex boundaries but were more sensitive to learning rate schedules.

- `tanh` generally outperformed `sigmoid` , particularly with annealed learning rates.

- Minibatch training (batch sizes 32–64) yielded faster convergence than full-batch.


## Directory Structure
Each file is a standlone — run it separately from terminal or in your IDE
