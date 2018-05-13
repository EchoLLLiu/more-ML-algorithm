import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

def loadDataSet(filename):
    '''加载数据'''
    X = []
    Y = []
    with open(filename, 'rb') as f:
        for idx, line in enumerate(f):
            line = line.decode('utf-8').strip()
            if not line:
                continue
                
            eles = line.split()
            if idx == 0:
                numFea = len(eles)
            eles = list(map(float, eles))
            
            X.append(eles[:-1])
            Y.append([eles[-1]])
    return np.array(X), np.array(Y)

def standarize(X):
    '''
    特征标准化处理：
    Args:
       X 样本集
    Returns:
      标准后的样本集
    '''
    m, n = X.shape
    values = {}  # 保存每一列的mean和std，便于对预测数据进行标准化
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        stdVal = features.std(axis=0)
        values[j] = [meanVal, stdVal]
        if stdVal != 0:
            X[:,j] = (features - meanVal) / stdVal
        else:
            X[:,j] = 0
    return X, values

def h(theta, X):
    '''定义函数模型'''
    return np.dot(X, theta)

def J(theta, X, Y):
    '''定义损失函数'''
    m = len(X)
    return np.sum(np.dot((h(theta, X) - Y).T, (h(theta, X) - Y)) / (2 * m))

def bgd(alpha, X, Y, maxloop, epsilon):
    '''定义梯度下降函数'''
    m, n = X.shape
    theta = np.zeros((n,1))  #初始化参数为0
    
    count = 0 # 记录迭代次数
    converged = False # 是否已收敛的标志
    cost = np.inf # 初始化代价值为无穷大
    costs = [J(theta, X, Y),] # 记录每一次的代价值
    
    thetas = {}  # 记录每一次参数的更新
    for i in range(n):
        thetas[i] = [theta[i,0],]
        
    while count <= maxloop:
        if converged:
            break
        count += 1
        
        # n个参数计算，并存入thetas中(循环单独计算theta)
        #for j in range(n):
        #    deriv = np.sum(np.dot(X[:,j].T, (h(theta, X) - Y))) / m
        #    thetas[j].append(theta[j,0] - alpha*deriv)
        # n个参数在当前theta中更新  
        #for j in range(n):
        #    theta[j,0] = thetas[j][-1]
        
        #同时计算theta
        theta = theta - alpha * 1.0 / m * np.dot(X.T, (h(theta, X) - Y))
        for j in range(n):
            thetas[j].append(theta[j,0])
            
        # 记录当前参数的函数代价，并存入costs
        cost = J(theta, X, Y)
        costs.append(cost)
            
        if abs(costs[-1] - costs[-2]) < epsilon:
            converged = True
    
    return theta, thetas, costs
