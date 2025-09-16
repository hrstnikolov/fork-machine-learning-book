You're absolutely right — and you've touched on a subtle but important distinction between **loss functions** and **update rules** in machine learning.

### Let's clarify:

#### ✅ **Loss Function (Global Perspective)**
In most machine learning models, a **loss function** is a mathematical expression that quantifies the error **over the entire dataset**. It's used to guide optimization, typically via gradient descent. Examples:
- Mean Squared Error (MSE)
- Cross-Entropy Loss

These are **aggregated** over all training examples, often averaged.

#### 🔁 **Perceptron Training (Local Updates)**
The **perceptron algorithm**, however, doesn't compute a global loss over all examples. Instead, it:
- Processes one example at a time.
- Checks if the prediction is correct.
- If incorrect, it updates the weights using the error:
  $$
  \mathbf{w} \leftarrow \mathbf{w} + y_i \mathbf{x}_i
  $$

This is a **local update rule**, not derived from minimizing a differentiable loss function.

### So is there a loss function?

In a formal sense, yes — researchers have defined a **perceptron loss** to analyze its behavior:
$$
L(\mathbf{w}) = \sum_{i=1}^{n} \max(0, -y_i (\mathbf{w}^\top \mathbf{x}_i))
$$

This is a **hinge-like loss**, only penalizing misclassified points. But the original algorithm doesn't minimize this explicitly — it just updates weights when misclassifications occur.

---

### Summary:
- **Original perceptron**: No explicit loss function used during training; updates are based on misclassification.
- **Modern interpretation**: You *can* define a loss function (like hinge loss) to analyze or extend the perceptron.

Would you like to see how this compares to something like Support Vector Machines (SVMs), which do use hinge loss explicitly?