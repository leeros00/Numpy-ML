import numpy as np
from typing import *

# TO DO: I needed to put this in its own standalone file, but will eventually
# integrate into the optimizer/utils

class LinearRegressor:
    def __init__(self) -> None:
        # TO DO: Add in toy dataset, other default hyperparameters
        pass

    def load_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        # TO DO: Consider just having this as a DataLoader object; integrate into the DataLoader class
        data = pd.read_csv(filename, header = 0)
        data = np.asarray(data)
        X = data[:,1:]
        Y = data[:,0].reshape(len(data[:,0]),1) # shape (N, 1)
        return (X, Y)

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        new_column = np.ones((X.shape[0], 1))
        design_matrix = np.concatenate((new_column, X), axis=1)
        return design_matrix

    def loss(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> float:
        Y_hat = X@theta
        mse = np.square(Y - Y_hat).mean()
        return mse/2
    
    def gradient(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        N, num_features = X.shape
        Y_hat = X@theta
        gradients = -1/N*np.sum((Y - Y_hat)*X, axis=0)
        gradients = np.reshape(gradients, (num_features, 1))
        return gradients

    def update(self, theta: np.ndarray, gradients: np.ndarray, lr: float) -> np.ndarray:
        return theta - lr*gradients
    
    def train(self, X_train: np.ndarray,
                    Y_train: np.ndarray,
                    theta0: np.ndarray,
                    num_epochs: int,
                    lr: float) -> np.ndarray:
        theta = theta0
        for i in range(num_epochs):
            grad = self.gradient(X_train, Y_train, theta)
            theta = self.update(theta, grad, lr)
        return theta
    
    def predict(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return X@theta

    def train_and_test(self, train_filename: str, test_filename: str, num_epochs: int, lr: float) -> Dict[str, Any]:
        X_train, Y_train = self.load_data(train_filename)
        X_test, Y_test = self.load_data(test_filename)

        design_X_train = self.design_matrix(X_train)
        design_X_test = self.design_matrix(X_test)

        theta0 = np.zeros((design_X_train.shape[1], 1))
        theta_final = self.train(design_X_train, Y_train, theta0, num_epochs, lr)

        train_predict = self.predict(design_X_train, theta_final)
        test_predict = self.predict(design_X_test, theta_final)

        train_error = self.loss(design_X_train, Y_train, theta_final)
        test_error = self.loss(design_X_test, Y_test, theta_final)
        
        return {'theta': theta_final,
                'train_error': train_error,
                'test_error': test_error}

        