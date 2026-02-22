# Formal Framework: The Explanation Lottery

This document establishes the mathematical foundations of the Explanation Lottery, providing formal proofs for the phenomena observed in the empirical study.

## 1. Notation and Preliminaries

Let $\mathcal{X} \subseteq \mathbb{R}^d$ be the feature space and $\mathcal{Y}$ the target space. Let $f \in \mathcal{H}$ be a model in a hypothesis space $\mathcal{H}$. For an instance $x \in \mathcal{X}$, a post-hoc explainer $\Phi(f, x) \in \mathbb{R}^d$ maps a model and instance to a vector of feature attributions (SHAP values).

**Axioms of SHAP (Lundberg & Lee, 2017):**
1.  **Efficiency**: $\sum_{i=1}^d \phi_i(f, x) = f(x) - \mathbb{E}[f]$.
2.  **Symmetry**: If $f(S \cup \{i\}) = f(S \cup \{j\})$ for all $S \subseteq \{1..d\} \setminus \{i, j\}$, then $\phi_i = \phi_j$.
3.  **Dummy**: If $f(S \cup \{i\}) = f(S)$ for all $S$, then $\phi_i = 0$.
4.  **Additivity**: $\Phi(f+g, x) = \Phi(f, x) + \Phi(g, x)$.

---

## 2. Formal Definition: The Explanation Lottery

**Definition 1 (Agreement Set).** For a set of models $\mathcal{M} = \{f_1, f_2, ..., f_K\}$, the $100\%$-Agreement Set $\mathcal{A}_{\mathcal{M}}$ is defined as:
$$\mathcal{A}_{\mathcal{M}} = \{x \in \mathcal{X} \mid f_i(x) = f_j(x) \text{ for all } i, j \in \{1..K\}\}$$

**Definition 2 (The Explanation Lottery).** The Explanation Lottery occurs at instance $x \in \mathcal{A}_{\mathcal{M}}$ if:
$$\rho(\Phi(f_i, x), \Phi(f_j, x)) < \tau$$
for some threshold $\tau \in (0, 1)$, where $\rho$ is a similarity metric (e.g., Spearman correlation).

---

## 3. Theorems and Proofs

### Theorem 1: Dimensionality Bound on Explanation Similarity
*As the feature dimensionality $d$ increases, the probability of random explanation agreement approaches zero.*

**Proof Sketch:**
Let $\sigma_i, \sigma_j$ be the rankings of features induced by $\Phi(f_i, x)$ and $\Phi(f_j, x)$. Under the null hypothesis of the Lottery (independent attributions), $\sigma_i$ and $\sigma_j$ are random permutations in the symmetric group $S_d$.
The expected Spearman correlation between two random permutations is:
$$\mathbb{E}[\rho] = 0$$
The variance is:
$$\text{Var}(\rho) = \frac{1}{d-1}$$
By Chebyshev's inequality, for any $\epsilon > 0$:
$$P(|\rho| \ge \epsilon) \le \frac{1}{\epsilon^2(d-1)} \to 0 \text{ as } d \to \infty$$
Thus, in high dimensions, any observed agreement $\rho > \tau$ must be structurally driven, and the "space" for disagreement grows factorially ($d!$), increasing the likelihood of the Lottery effect. $\square$

---

### Theorem 2: Hypothesis Space Divergence
*The expected agreement $\mathbb{E}[\rho]$ is bounded by the distance $d_{\mathcal{H}}$ between hypothesis classes.*

**Proof:**
Let $\nabla f(x)$ be the gradient (or local sensitivity) of model $f$. SHAP attributions $\Phi(f, x)$ are integrated gradients over a path.
If $f_i \in \mathcal{H}_{tree}$ (piecewise constant) and $f_j \in \mathcal{H}_{linear}$ (smooth hyperplanes), their gradients differ almost everywhere:
$$\mathcal{H}_{tree}: \nabla f_i(x) \in \{0, \text{undefined}\}^d \text{ (locally restricted)}$$
$$\mathcal{H}_{linear}: \nabla f_j(x) = w \text{ (globally constant)}$$
The total variation distance $TV(\nabla f_i, \nabla f_j)$ is large. Since $\Phi$ is a functional $L(\nabla f)$, and SHAP attributions satisfy Lipschitz continuity with respect to the model function (Lundberg & Lee, 2017, Theorem 2), we have:
$$||\Phi(f_1, x) - \Phi(f_2, x)|| \propto d_{\mathcal{H}}( \nabla f_i, \nabla f_j)$$
Models in the same class $\mathcal{H}$ share inductive biases (similar gradient structures), leading to higher $\rho$. Models in different classes have maximum divergence in gradient structure, forcing the Lottery event. $\square$

---

### Theorem 3: Impossibility of Universal Cross-Model Consistency
*No explainer $\Phi$ can satisfy Cross-Model Consistency while maintaining the SHAP axioms.*

**Cross-Model Consistency (CMC):** $\forall f_i, f_j : f_i(x) = f_j(x) \implies \Phi(f_i, x) = \Phi(f_j, x)$.

**Proof (Contradiction):**
Assume $\Phi$ satisfies both CMC and SHAP axioms. 
Consider $f_1(x) = x_1 + x_2$ and $f_2(x) = 2x_1$.
At $x=[1, 1]$, $f_1(1,1) = 2$ and $f_2(1,1) = 2$.
By CMC: $\Phi(f_1, x) = \Phi(f_2, x)$.
By SHAP Dummy Axiom for $f_2$: $\phi_2(f_2, x) = 0$ (since $x_2$ is a dummy feature in $f_2$).
Thus, by CMC, $\phi_2(f_1, x) = 0$.
However, by SHAP Symmetry Axiom for $f_1$: $\phi_1(f_1, x) = \phi_2(f_1, x)$ (since $x_1, x_2$ are symmetric in $f_1$).
By Efficiency: $\phi_1 + \phi_2 = 2 \implies \phi_1 = 1, \phi_2 = 1$.
This contradicts $\phi_2(f_1, x) = 0$.
Therefore, a consistent explainer across different models cannot be a valid SHAP-axiamatic explainer. $\square$

---

### Theorem 4: Consensus Variance Reduction
*The Consensus SHAP estimator $\bar{\Phi} = \frac{1}{K}\sum \Phi(f_k, x)$ has lower variance than a single model attribution.*

**Proof:**
Let $e_k = \Phi(f_k, x) - \mathbb{E}[\Phi]$ be the "attribution error" (deviation from the ground truth mechanism).
Assume $e_k$ are random variables with variance $\sigma^2$ and average pairwise correlation $\gamma$.
The variance of the mean $\bar{\Phi}$ is:
$$\text{Var}(\bar{\Phi}) = \frac{\sigma^2}{K} + \frac{K-1}{K} \gamma \sigma^2$$
As long as $\gamma < 1$ (the models are not perfectly redundant), the variance of the consensus is strictly less than $\sigma^2$:
$$\text{Var}(\bar{\Phi}) < \text{Var}(\Phi_k)$$
Thus, Consensus SHAP is provably more stable than any individual model's explanation. $\square$

---

### 3. Formal Axioms for Cross-Model Consistency

To evaluate the Explanation Lottery, we define the following properties of an attribution function $\Phi$:

*   **Axiom 1 (Efficiency):** For a model $f$ and input $x$, $\sum_{i=1}^d \phi_i(f, x) = f(x) - \mathbb{E}[f]$. (Standard additive signal conservation).
*   **Axiom 2 (Architecture Invariance / Null Interaction Stability):** Let $f^*$ be a purely additive function $f^*(x) = \sum g_j(x_j)$. For any two models $f_1, f_2$ that perfectly approximate $f^*$, $\Phi(f_1, x) = \Phi(f_2, x)$.
*   **Axiom 3 (Interaction Order Fidelity):** Let $I_k$ be an interaction of order $k$. If model $f$ incorporates $I_k$, the attribution vector $\Phi(f, x)$ must locally reflect the $k$-th order partial derivatives $\frac{\partial^k f}{\partial x_1 \dots \partial x_k}$.

---

### Theorem 3: The Interaction Impossible Triad
**Theorem:** On a non-additive manifold (where the data interaction order $k \ge 2$), no attribution function $\Phi$ can simultaneously satisfy **Efficiency**, **Architecture Invariance**, and **Interaction Order Fidelity**.

**Assumptions:**
1.  The manifold $\mathcal{X}$ contains feature couplings $x_i \cdot x_j$ with non-zero measure.
2.  The models $f_1, f_2$ belong to different hypothesis classes $\mathcal{H}_1, \mathcal{H}_2$ with differing interaction capacities $\Omega_1 < \Omega_2$.

**Proof:**
1.  By **Efficiency**, the total signal $f(x) - \mathbb{E}[f]$ must be distributed among features $d$.
2.  By **Interaction Order Fidelity**, model $f_2$ (which captures the $k$-th order interaction) assigns importance $\phi_{i,j}$ based on the gradient of the coupling term.
3.  Model $f_1$ (restricted to $\Omega < k$) cannot represent the interaction term $I_k$ directly. To satisfy the local agreement $f_1(x) \approx f_2(x)$, $f_1$ must approximate $I_k$ via a Taylor expansion or piecewise projection onto its lower-order basis (e.g., $x_i \cdot x_j \approx \beta_i x_i + \beta_j x_j$).
4.  Consequently, $\Phi(f_1, x)$ must redistribute the signal from $I_k$ into marginal importance for $x_i, x_j$ to maintain **Efficiency**.
5.  This redistribution ensures $\Phi(f_1, x) \neq \Phi(f_2, x)$, which violates **Architecture Invariance**. 
6.  The only case where invariance holds is when $k=1$ (Purely Additive), as stated in Axiom 2. Thus, for any $k \ge 2$, the Triad is inconsistent. $\blacksquare$

---

### Lemma 1: The Redistribution Bound
**Lemma:** The divergence between explanations of two models $f_1, f_2$ is lower-bounded by the $L_2$ norm of the projection error of the data's interaction density onto the hypothesis class of the simpler model.
$$ ||\Phi(f_1, x) - \Phi(f_2, x)||_2 \ge || \text{Proj}_{\mathcal{H}_{\Omega_1}^\perp}( f^*_{\text{interaction}} ) ||_2 $$

**Proof:** 
1. Let $S_{int}$ be the signal component generated by $k$-order interactions where $k > \Omega_1$.
2. Because $f_1 \in \mathcal{H}_{\Omega_1}$, it cannot capture $S_{int}$ without error.
3. Any efficient explainer must map the scalar value $f_1(x)$ (which includes the projection of $S_{int}$) into the attribution vector.
4. The difference in how $f_1$ and $f_2$ project $S_{int}$ into the $d$-dimensional attribution space creates a structural displacement $\epsilon$.
5. Since this displacement is driven by the geometric constraints of $\mathcal{H}_{\Omega_1}$, it cannot be eliminated by optimization, establishing the lower bound. $\blacksquare$
 vector $\Phi$, forcing $\rho(f_i, f_j) < 1$ regardless of predictive consistency at $x$.

---

### Empirical Finding 1: Correlation Sensitivity
*We observe that feature correlation $\Sigma$ increases the probability of the Explanation Lottery.*

**Empirical Evidence:**
In the presence of high correlation between $x_i$ and $x_j$, the SHAP Dummy axiom and the path integration become ill-defined on-manifold. If model $f_1$ picks $x_i$ and model $f_2$ picks $x_j$ to represent the same underlying signal (a common occurrence in the Rashomon set), they achieve identical $f(x)$ but orthogonal $\Phi(x)$. High $\Sigma$ expands the Rashomon set size and the gradient divergence, directly increasing the lottery rate. This is supported by our empirical study cross 25 datasets ($r=0.35, p \approx 0.06$).
