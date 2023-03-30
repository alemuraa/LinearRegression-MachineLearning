import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

yes = True
no = False

df1=pd.read_csv("turkish-se-SP500vsMSCI.csv", header=None)
df2=pd.read_csv("mtcarsdata-4features.csv")

def create_vector(df, col1, col2):
    x = []
    y = []

    for k in range(df.shape[0]):
        x.append(df.iloc[k][col1])
        y.append(df.iloc[k][col2])
    return(x, y)

def estimate_coef_without_intercept(x, y):
    num_w = 0
    den_w = 0

    for k in range(len(x)):
        num_w += x[k]*y[k]
        den_w += x[k]**2

    w = num_w / den_w
    return(w)

def plot_regression_line_without_intercept(x, y, w):
    y_pred = []

    for k in range(len(x)): 
        y_pred.append(w*x[k])

    plt.scatter(x, y, color = RandomColor(), s=5) 
    plt.plot(x, y_pred, color = RandomColor())
    plt.legend(["Data", "Regression Line"], loc ="lower right")

def estimate_coef_with_intercept(x, y):
    m_x = np.mean(x)
    m_y = np.mean(y)
    num_w_1 = 0
    den_w_1 = 0

    for k in range(len(x)):
        num_w_1 += (x[k] - m_x)*(y[k] - m_y)
        den_w_1 +=(x[k] - m_x)**2

    w_1 = num_w_1 / den_w_1
    w_0 = m_y - w_1 * m_x
    return(w_0, w_1)

def plot_regression_line_with_intercept(x, y, w):
    y_pred = []
    
    for k in range(len(x)): 
        y_pred.append(w[0]+w[1]*x[k])
    
    plt.scatter(x, y, color = RandomColor(), s=5)
    plt.plot(x, y_pred, color = RandomColor())
    plt.legend(["Data", "Regression Line"], loc ="lower right")

def randomSubset(n):
    df = df1.sample(frac=n)
    x, y = create_vector(df, 0, 1)
    w = estimate_coef_without_intercept(x, y)
    plot_regression_line_without_intercept(x, y, w)
    plt.xlabel("X")
    plt.ylabel("Y")

def MultiDimensional(x, y):
    x_t = np.transpose(x)
    x_p = np.dot(x_t, x)

    try:
        x_i = np.linalg.inv(x_p)
    except(np.linalg.LinAlgError):
        y=0
        return(y)

    x_p = np.dot(x_i, x_t)
    w = np.dot(x_p, y)
    y = np.dot(x, w)
    return(y)

def OneDimensionalMeanSquareError(x, y, w):
    y_pred = []

    if type(w) == np.float64 or type(w) == int: 
        for k in range(len(x)): 
            y_pred.append(w*x[k])
    else: # w belongs to class 'tuple'
        for k in range(len(x)): 
            y_pred.append(w[0]+w[1]*x[k])
    
    sum = 0
    for n in range(len(y)):
        sum += (y_pred[n] - y[n])**2
    
    jmse = sum/len(y)
    return(jmse)

def MultiDimensionalMeanSquareError(y, t):
    op = y - t
    norm = np.linalg.norm(op)
    pot = pow(norm, 2)
    jmse = pot/2
    return(jmse)

def BuildGraph(df, col1, col2, c):
    x, y = create_vector(df, col1, col2)
    
    if c:
        w = estimate_coef_without_intercept(x, y)
        j = OneDimensionalMeanSquareError(x, y, w)
        plot_regression_line_without_intercept(x, y, w)
    else:
        w = estimate_coef_with_intercept(x, y)
        j = OneDimensionalMeanSquareError(x, y, w)
        plot_regression_line_with_intercept(x, y, w)

    return(j)

def MultidimensionalInit(df, c):
    x = df[df.columns[2:5]].values
    y = df[df.columns[1]].values
    if c: print('Mpg Predicted: ', MultiDimensional(x_4, y_4),'\n')
    j = MultiDimensionalMeanSquareError(MultiDimensional(x, y), y)
    return(j)

def OneDimensionalTrainTestHist(df, n, col1, col2, c):
    a = []
    b = []

    for i in range(n):
        train = df.sample(frac=r)
        test = df.drop(train.index, axis = 0)
        x_train, y_train = create_vector(train, col1, col2)
        x_test, y_test = create_vector(test, col1, col2)

        if c:
            w_train = estimate_coef_without_intercept(x_train, y_train)
            w_test = estimate_coef_without_intercept(x_test, y_test)
        else:
            w_train = estimate_coef_with_intercept(x_train, y_train)
            w_test = estimate_coef_with_intercept(x_test, y_test)

        j_train = OneDimensionalMeanSquareError(x_train, y_train, w_train)
        j_test = OneDimensionalMeanSquareError(x_test, y_test, w_test)
        
        a.append(j_train)
        b.append(j_test)

    return a, b

def MultiDimensionalTrainTestHist(df, n):
    a =[]
    b = []

    for i in range(n):
        train = df.sample(frac=r)
        test = df.drop(train.index, axis = 0)
        x_train = train[train.columns[2:5]].values
        y_train = train[train.columns[1]].values
        x_test = test[test.columns[2:5]].values
        y_test = test[test.columns[1]].values
        j_train = MultiDimensionalMeanSquareError(MultiDimensional(x_train, y_train), y_train)
        j_test = MultiDimensionalMeanSquareError(MultiDimensional(x_test, y_test), y_test)
        a.append(j_train)
        b.append(j_test)

    return a, b

def RandomColor():
    color = "#"+''.join([rd.choice('0123456789ABCDEF') for i in range(6)])
    return(color)

# One-dimensional problem without intercept on the Turkish stock exchange data
title = 'One-dimensional problem without intercept on Df1'
BuildGraph(df1, 0, 1, yes)
plt.title(title)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Compare graphically the solution obtained on different random subsets (10%) of the whole data set
n = 0.1
for i in range(4):
    randomSubset(n)
title = 'One-dimensional Df1 with 10% of random subsets Comparison'
plt.title(title)
plt.show()

# One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight
BuildGraph(df2, 4, 1, no)
title = 'One-dimensional problem with intercept on Df2'
plt.title(title)
plt.xlabel("mpg")
plt.ylabel("weight")
plt.show()

# Multi-dimensional problem on the complete MTcars data, using all four columns (predict mpg with the other three columns)
x_4 = df2[df2.columns[2:5]].values
y_4 = df2[df2.columns[1]].values
print('mpg predicted: ', MultiDimensional(x_4, y_4),'\n')

# Re-run 1,3 and 4 from task 2 using only 5% of the data and the remaining 95% and compute the objective (mean square error)
n = 0.05
df1_5 = df1.sample(frac=n)
df1_95 = df1.drop(df1_5.index, axis = 0)
j1_5 = BuildGraph(df1_5, 0, 1, yes)
j1_95 = BuildGraph(df1_95, 0, 1, yes)
title = 'One-dimensional problem without intercept on Df1 with 5% and 95% of the data'
plt.title(title)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
print('Jmse(1)[5%] = ', j1_5)
print('Jmse(1)[95%] = ', j1_95, '\n')

df2_1_5 = df2.sample(frac=n)
df2_1_95 = df2.drop(df2_1_5.index, axis = 0)
j3_5 = BuildGraph(df2_1_5, 4, 1, no)
j3_95 = BuildGraph(df2_1_95, 4, 1, no)
title = 'One-dimensional problem with intercept on Df2 with 5% and 95% of the data'
plt.title(title)
plt.xlabel("mpg")
plt.ylabel("weight")
plt.show()
print('Jmse(3)[5%] = ', j3_5)
print('Jmse(3)[95%] = ', j3_95, '\n')

df2_2_5 = df2.sample(frac=n)
df2_2_95 = df2.drop(df2_2_5.index, axis = 0)
j4_5 = MultidimensionalInit(df2_2_5, yes)
j4_95 = MultidimensionalInit(df2_2_95, yes)
print('Jmse(4)[5%] = ', j4_5)
print('Jmse(4)[95%] = ', j4_95, '\n')

# Repeat for different training-test random splits
r = rd.random()
perc_train = int(r*100)
n = 100
train1 , test1 = OneDimensionalTrainTestHist(df1, n, 0, 1, yes)
train3 , test3 = OneDimensionalTrainTestHist(df2, n, 4, 1, no)
train4 , test4 = MultiDimensionalTrainTestHist(df2, n)
p = [train1, test1, train3, test3, train4, test4]

for i in range(1, 7):
    plt.subplot(2,3,i)
    plt.hist(p[i-1])

print('MSE Train = '+ str(perc_train)+'% of data, MSE Test = '+str(100-perc_train)+'% of data', '\n')
plt.show()