# KNN classifer 
# 1. load training data 
# 2. predict testing data by measureing the distance 
import numpy as np 
class KNN: 
    def __init__(self, k): 
        self.k = k 
    
    def train(self, X, y): 
        self.X = X 
        self.y = y 

    def predict(self, testX): 
        distances = np.sum(testX * testX, axis=1, keepdims=True) + 2 * testX.dot(self.X.T) + np.sum(self.X * self.X, axis=1, keepdims=True).T
        m = testX.shape[0]
        pred = np.zeros(m, dtype=int)
        for i in range(m): 
            cands = np.argsort(distances[i])[:self.k]
            counts = np.bincount(self.y[cands])
            pred[i] = np.argmax(counts)
        return pred 

if __name__ == "__main__": 
    from sklearn.datasets import load_iris 
    X, y = load_iris(return_X_y=True)
    knn = KNN(5)
    knn.train(X, y)
    m = X.shape[0]
    idxs = np.random.choice([False, True], size=m, p=[0.2, 0.8])
    testX, testy = X[idxs], y[idxs]
    pred = knn.predict(testX)
    size = testX.shape[0]
    print(np.sum(pred == testy))
    acc = np.sum(pred == testy) / size 
    print(acc)
