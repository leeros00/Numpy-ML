import numpy as np
from typing import *

class DataLoader:
    # TO DO: Work on this. Make it kind of like PyTorch.

    pass

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X: List[str] = []
    Y: List[str] = []

    attribute_names: List[str] = []
    with open(filename, 'r') as f: 
        reader = csv.reader(f)

        # get attribute list, get rid of label header
        attribute_names = next(reader)[:-1] 

        # get X, Y values and convert them to numpy arrays
        for row in reader: 
            X.append(row[:-1])
            Y.append(row[-1])
        X = np.array(X)
        Y = np.array(Y)

    return (X, Y, attribute_names)

class DecisionStump:
    def __init__(self) -> None:
        pass

    def majority_vote(self, X: np.ndarray, Y: np.ndarray) -> str:
        label_count = Dict[str, str] = dict()
        for label in Y:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1

        max_label, max_count = "", 0
        for label, count in label_count.items()
        if count > max_count:
            max_label = label
            max_count = count

        return str(max_label)

    def split(self, X: np.ndarray, Y: np.ndarray, split_attribute: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = sorted(list(set([val for val in X[:, split_attribute]])))
        left_val, right_val = values[0], values[1]

        X_left, Y_left, X_right, Y_right = [], [], [], []
        for X_instance, Y_instance in zip(X, Y):
            if X_instance[split_attribute] == left_val:
                X_left.append(X_instance)
                Y_left.append(Y_instance)
            else:
                X_right.append(X_instance)
                Y_right.append(Y_instance)
        return np.array(X_left), np.array(Y_left), np.array(X_right), np.array(Y_right)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, attribute_index: int) -> Tuple:
        X_left, Y_left, X_right, Y_right = self.split(X=X_train, Y=Y_train, split_attribute=attribute_index)
        left_label = self.majority_vote(X_left, Y_left)
        right_label = self.majority_vote(X_right, Y_right)
        return left_label, right_label

    def predict(self, left_label: str, right_label: str, X: np.ndarray, attribute_index: int) -> np.ndarray:
        values = sorted(list(set([val for val in X[:, attribute_index]])))
        left_val, right_val = values[0], values[1]

        Y_pred = []
        for instance in X:
            if instance[attribute_index] == left_val
                Y_pred.append(left_label)
            else:
                Y_pred.append(right_label)
        return np.array(Y_pred)

    def error_rate(self, Y: np.ndarray, Y_pred: np.ndarray) -> float:
        incorrect: int = 0
        total: int = Y.shape[0]
        for y, y_pred in zip(Y, Y_pred):
            if y != y_pred:
                incorrect += 1
        return incorrect/total

    def train_and_test(self, train_filename: str, test_filename: str, attribute_index: str) -> Dict[str, Any]:
        X_train, Y_train, attribute_names = load_data(train_filename)
        X_test, Y_test, _ = load_data(test_filename)

        left_label, right_label = self.train(X_train=X_train, Y_train=Y_train, attribute_index=attribute_index)
        Y_pred_train = self.predict(left_label, right_label, X_train, attribute_index)
        Y_pred_test = self.predict(left_label, right_label, X_test, attribute_index)

        train_error_rate = self.error_rate(Y_train, Y_pred_train)
        test_error_rate = self.error_rate(Y_test, Y_pred_test)

        return {'attribute_names' : attribute_names,
                'stump': (left_label, right_label), 
                'train_error_rate': train_error_rate, 
                'test_error_rate' : test_error_rate}

class Node:
    def __init__(self, X: np.ndarray, Y: np.ndarray, attribute_names: List[str], label_names: List[str]):
        self.X: np.ndarray = X 
        self.Y: np.ndarray = Y 
        self.attribute_names: List[str] = attribute_names 
        self.label_names: List[str] = label_names 
        
        self.left: Node = None
        self.right: Node = None        
        self.split_index: int = None   
        self.predicted_label: str = ""



class DecisionTree(DecisionStump):
    def __init__(self) -> None:
        
        #super().__init__()
        pass

    def tree_print(self, current_node: Node, current_depth: int=0) -> None:
        if current_node == None:
            return None
        else:
            feat_names, feat_counts = np.unique(current_node.X, return_counts=True)
            label_0_count = sum([i==current_node.label_names[0] for i in current_node.Y])
            label_1_count = sum([i==current_node.label_names[1] for i in current_node.Y])

            if current_depth == 0:
                print('[' + str(label_0_count) + '/' + str(current_node.label_names[0]) + 
                ' ' + str(label_1_count) + '/' + str(current_node.label_names[1]) + ']')

                current_depth = current_depth + 1

                self.tree_print(current_node.left, current_depth)
                self.tree_print(current_node.right, current_depth)
            else:
                split_val = current_node.X[0,current_node.split_index]
                split_name = current_node.attribute_names[current_node.split_index] 

                print("|  "*current_depth, end = "")
                print(str(split_name) + ' = ' + str(split_val) + ': ' +
                '[' + str(label_0_count) + ' ' + str(current_node.label_names[0]) + 
                '/' + str(label_1_count) + ' ' + str(current_node.label_names[1]) + ']')
                current_depth = current_depth + 1
                self.tree_print(current_node.left, current_depth)
                self.tree_print(current_node.right, current_depth)

    def entropy(self, Y: np.ndarray) -> float:
        _, label_counts = np.unique(Y, return_counts=True)
        N = Y.shape[0]
        H_Y: float = 0.0
        for N_y in label_counts:
            prob = N_y/N
            H_Y += -1*(prob)*np.log2(prob)
        return H_Y

    def conditional_entropy(self, X_m: np.ndarray, Y: np.ndarray) -> float:
        N = X_m.shape[0]
        X_m_values, X_m_counts = np.unique(X_m, return_counts=True)
        H_Y_X = 0.0
        for i, x in enumerate(X_m_values):
            # construct X,Y dataset for this attribute value x
            Y_X_mx = np.array([Y[i] for i in range(N) if X_m[i] == x])

            # compute specific conditional entropy
            H_Y_given_X = self.entropy(Y_X_mx)
            H_Y_X += X_m_counts[i]/N * H_Y_given_X
        return H_Y_X

    def mutual_information(self, X_m: np.ndarray, Y: np.ndarray) -> float:

        H_Y = self.entropy(Y)
        H_Y_given_X = self.conditional_entropy(X_m, Y)

        return H_Y - H_Y_given_X

    def find_best_attribute(self, X: np.ndarray, Y: np.ndarray) -> int:
        best_attribute: int = 0
        max_mi: float = 0.0
        for m, X_m in enumerate(X.T):
            cur_mi = self.mutual_information(X_m, Y)
            if cur_mi > max_mi:
                max_mi = cur_mi
                best_attribute = m
        return best_attribute

    def train(self, X: np.ndarray, Y: np.ndarray, attribute_names: List[str], max_depth: int) -> Node:
        N, M = X.shape
        if max_depth > M:
            max_depth = M
        label_names = np.unique(Y).tolist()
        root = Node(X=X, Y=Y, attribute_names=attribute_names, label_names=label_names)

        return self.train_tree(node=root, depth=0, max_depth=max_depth)

    def train_tree(self, node: Node, depth: int, max_depth: int) -> Node:
        if depth >= max_depth:
            node.predicted_label = self.majority_vote(X=node.X, Y=node.Y)
            return node
        elif len(np.unique(node.Y)) == 1:
            node.predicted_label = node.Y[0]
            return node
        else:
            best_splitter = self.find_best_attribute(X=node.X, Y=node.Y)
            if len(np.unique(node.X[:, best_splitter])) < 2:
                node.predicted_label = self.majority_vote(X=node.X, Y=node.Y)
                return node
        
        X_left, Y_left, X_right, Y_right = self.split(X=node.X, Y=node.Y, split_attribute=best_splitter)

        left_node = Node(X=X_left,
                         Y=Y_left,
                         attribute_names=node.attribute_names,
                         label_names=node.label_names)
        left_node.split_index = best_splitter

        right_node = Node(X=X_right,
                          Y=Y_right,
                          attribute_names=node.attribute_names,
                          label_names=node.label_names)
        right_node.split_index = best_splitter

        node.left = self.train_tree(node=left_node, depth=depth+1, max_depth=max_depth)
        node.right = self.train_tree(node=right_node, depth=depth+1, max_depth=max_depth)

        return node

    def traverse_tree(self, node: Node, x: np.ndarray) -> str:
        if node.left == None and node.right == None:
            return node.predicted_label
        else:
            split_attribute = node.left.split_index
            x_m = x[split_attribute]
            x_m_left = node.left.X[0, split_attribute]
            if x_m == x_m_left:
                return self.traverse_tree(node.left, x=x)
            else:
                return self.traverse_tree(node=node.right, x=x)

    def predict(self, trained_tree: Node, X_test: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
        N = Y_test.shape[0]
        all_labels = []
        for i in range(N):
            predicted_label = self.traverse_tree(node=trained_tree, x=X_test[i])
            all_labels = np.append(predicted_label)
        return np.array(all_labels)

    def train_and_test(self, train_filename: str, test_filename: str, max_depth: str) -> Dict[str, Any]:
        # TO DO: Consider using a super()
        X_train, y_train, attribute_names = load_data(train_filename)
        X_test, y_test, attribute_names = load_data(test_filename)

        trained_tree = self.train(X_train, y_train, attribute_names, max_depth)
        y_pred_train = self.predict(trained_tree, X_train, y_train)
        y_pred_test = self.predict(trained_tree, X_test, y_test)

        train_error = self.error_rate(y_pred_train, y_train)
        test_error = self.error_rate(y_pred_train, y_test)

        return {'tree': trained_tree, 'train_error': train_error, 'test_error': test_error}

    


