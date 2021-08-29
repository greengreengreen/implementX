import numpy as np 

class LogisticRegression: 
    def __init__(self) -> None:
        self.X = None 
        self.W = None 
        self.b = None 
        self.loss = None 
        self.y = None 

    def sigmoid(self, z): 
        # z = np.maximum(-10, z) as a cap 
        # z = np.minimum(30, z)
        # import warnings 
        # with warnings.catch_warnings(record=True) as w: 
            # print(max(z))
            # print(min(z))
        return 1/(1 + np.exp(-z))

    def getLoss(self): 
        z = self.X.dot(self.W) + self.b 
        eps = 1e-5
        probs = self.sigmoid(z)
        m = self.X.shape[0]
        loss = -np.sum(self.y * np.log(probs + eps) + (1 - self.y) * np.log(1-probs + eps))/m
        return loss 

    def train(self, X, y, method="gradient_descent", learning_rate=0.01): 
        # initialize 
        m, n = X.shape
        self.X, self.y = X, y 
        self.W = np.random.rand(n, 1) 
        self.b = 0 

        if method == "gradient_descent": 
            prevloss = float("inf")
            curloss = self.getLoss()
            iter, eps = 0, 1e-4
            loss = []
            while prevloss - curloss > eps and iter <= 5000: 
                loss.append(curloss)
                z = self.X.dot(self.W) + self.b 
                probs = self.sigmoid(z)
                dW = -self.X.T.dot(self.y - probs)/m
                db = -np.sum(self.y - probs)/m
                self.W -= learning_rate * dW 
                self.b -= learning_rate * db 
                prevloss = curloss 
                curloss = self.getLoss()
                # print("At iter %d using gradient descent, the loss is %f" % (iter, curloss))
                iter += 1 

        if method == "newton_method": 
            prevloss = float("inf") 
            curloss = self.getLoss()
            iter, eps = 0, 1e-4
            loss = []
            while prevloss - curloss > eps and iter <= 5000: 
                loss.append(curloss)
                z = self.X.dot(self.W) + self.b 
                probs = self.sigmoid(z)
                dW = -self.X.T.dot(self.y - probs) / m 
                db = -np.sum(self.y - probs) / m

                W = np.diag(np.squeeze(probs * (1-probs)))
                hessionW = self.X.T.dot(W).dot(self.X) / m 
                hessionb = np.sum(W) / m 

                inv_hessionW = np.linalg.inv(hessionW)
                inv_hessionb = 1/ hessionb

                self.W = self.W - learning_rate * inv_hessionW.dot(dW)
                self.b = self.b - learning_rate * inv_hessionb * db

                prevloss = curloss 
                curloss = self.getLoss()
                # print("At iter %d using Newton Method, the loss is %f" % (iter, curloss))
                iter += 1 
        return (loss, self.W, self.b)

    def predict(self, X, W, b): 
        z = X.dot(W) + b 
        probs = self.sigmoid(z)
        pred = probs > 0.5 
        return pred 

if __name__ == "__main__": 
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    from sklearn import preprocessing 
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    y = np.expand_dims(y, axis=1)
    
    lr_gradient = LogisticRegression()
    gradient_loss, W, b = lr_gradient.train(X, y)

    lr_newton = LogisticRegression()
    newton_loss, W, b = lr_newton.train(X, y, "newton_method")

    import matplotlib.pyplot as plt 
    plt.plot(np.arange(len(newton_loss)), newton_loss, label="Newton Loss")
    plt.plot(np.arange(len(gradient_loss)), gradient_loss, label="Gradient Loss")
    plt.xlim([0, 1000])
    plt.legend()
    plt.show()

    # pred = lr.predict(X, W, b)
    # n = X.shape[0] 
    # acc = np.sum(pred == y) / n 
    # print("acc: %f"%acc)

