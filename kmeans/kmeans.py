# kmeans steps 
# 1. initialization: initialize each data point into a random cluster 
# 2. Repeat until convergence: 
#    1) calculate the center of each cluster 
#    2) re-assign each point to the new cluster. 

import numpy as np 

class Kmeans: 
    def __init__(self, X, k) -> None:
        self.X = X 
        self.k = k 
        # step1: random assign 
        self.m = self.X.shape[0] # the number of training data 
        self.n = self.X.shape[1] # the number of features 
        self.assignments = np.random.choice(k, size=self.m)
        self.clusters = np.zeros((self.k, self.n), dtype=float)
    
    def get_distance(self): 
        # calculate the sum of within cluster distance 
        distance = np.linalg.norm(self.X - self.clusters[self.assignments])
        return np.sum(distance)

    def assign(self): 
        # assign each point to the nearest cluster 
        for i in range(self.m): 
            distance = np.linalg.norm(self.X[i] - self.clusters, axis=1)
            self.assignments[i] = np.argmin(distance) 

    def update_cluster(self): 
        # update each cluster to be the mean of all the points assigned 
        for c in range(self.k): 
            idx = self.assignments == c
            self.clusters[c,:] = self.X[idx].mean(axis=0)

    def train(self): 
        # train 
        self.update_cluster()
        prev_distance = self.get_distance()
        cur_distance = float("inf")
        eps = 1e-5
        print("start_training")
        iter = 0 
        while abs(cur_distance - prev_distance) > eps: 
            self.assign()
            self.update_cluster()
            prev_distance = cur_distance
            cur_distance = self.get_distance()
            print("At iter %d, the distance is %f" % (iter, cur_distance))
            iter += 1

    def predict(self, testX): 
        n = testX.shape[0]
        assignments = np.zeros(n)
        for i in range(n): 
            distances = np.linalg.norm(testX[i] - self.clusters, axis=1)
            assignments[i] = np.argmin(distances)
        return assignments

# test the above functions 
if __name__ == "__main__": 
    from sklearn.datasets import make_blobs 
    X, y = make_blobs(n_samples= 600, centers = 5, n_features = 2, random_state = 0)
    kmeans = Kmeans(X, k=5)
    kmeans.train()
    res = kmeans.predict(X)
    print(res)

