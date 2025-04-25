from clearml import Task, Logger
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset once
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# -------- Task 1: Random Forest --------
task_rf = Task.init(project_name="ClearML Demo", task_name="Iris - Random Forest")
logger_rf = task_rf.get_logger()

clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf.fit(X_train, y_train)
preds_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, preds_rf)

print(f"[Random Forest] Accuracy: {acc_rf}")
logger_rf.report_scalar("accuracy", "random_forest", iteration=0, value=acc_rf)

# -------- Task 2: Decision Tree --------
task_dt = Task.create(project_name="ClearML Demo", task_name="Iris - Decision Tree")
task_dt.connect({})
task_dt.mark_started()
logger_dt = task_dt.get_logger()

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)
preds_dt = clf_dt.predict(X_test)
acc_dt = accuracy_score(y_test, preds_dt)

print(f"[Decision Tree] Accuracy: {acc_dt}")
logger_dt.report_scalar("accuracy", "decision_tree", iteration=0, value=acc_dt)

task_dt.mark_completed()
