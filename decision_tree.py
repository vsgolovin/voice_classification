from pathlib import Path
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ft_export import get_array_paths


# load train dataset
path_X, path_y = get_array_paths("train")
X_train = np.loadtxt(path_X)
y_train = np.loadtxt(path_y)

# load dev (validation) dataset
path_X, path_y = get_array_paths("dev")
X_dev = np.loadtxt(path_X)
y_dev = np.loadtxt(path_y)

# fit decision tree
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_dev)
acc = sum(predictions == y_dev) / len(y_dev)
print(f"Decision tree classification accuracy: {(acc*100):.1f}%")

with open(Path("models/dtree.pkl"), "wb") as outfile:
    pickle.dump(clf, outfile)
