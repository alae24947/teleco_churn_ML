1️⃣ Model Choice: Logistic Regression
We used LogisticRegression(max_iter=1000) because it’s simple, fast, and interpretable.

Coefficients are directly usable for business insights.

Probabilities let us adjust the decision threshold to maximize F1.

max_iter=1000 ensures convergence.

C=1 (default) provides standard regularization, enough for a first approach.

2️⃣ Model Evaluation

2.1 Default threshold (0.5)

Predict ≥0.5 → Churn, else non-churn.

Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC.

Dataset is slightly imbalanced, so 0.5 is not necessarily optimal.

Results: predicts non-churners well (class 0), but misses many churners (recall 0.56). Accuracy = 0.81, AUC = 0.842.

2.2 Exploratory threshold optimization

We tested thresholds from 0.01 to 0.99 to maximize F1 on the test set.

Optimal threshold balances false positives and false negatives.

⚠ Note: using the test set makes results slightly optimistic; ideally, a separate validation set is needed.

2.3 Optimized threshold results

Churner recall improves (0.78), precision drops (0.51), accuracy decreases (0.75), AUC unchanged.

Takeaway: better at catching at-risk clients, at the cost of more false positives—a common trade-off in churn.

3️⃣ Graphical Analysis

Confusion Matrix: visualizes false positives/negatives.

ROC Curve: evaluates separability independent of threshold.

Precision-Recall Curve: better for imbalanced classes.

F1 vs Threshold: shows why the optimal threshold differs from 0.5.

4️⃣ Coefficient Analysis

Month-to-month contracts → higher churn.

Longer tenure → lower churn.

Higher monthly charges → higher churn.

Business insight: helps target at-risk clients and plan retention strategies.
