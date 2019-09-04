import numpy as np


def sampling_importance_resample(X, f, n):
    """
    重要性抽样
    :param X:样本集
    :param f:样本集的概率分布函数
    :param n:抽样个数
    :return:抽样结果
    """
    N = len(X)
    sampling_n = n*10 if n*10 < N else N
    candidate = np.random.choice(X, sampling_n)
    w_ = np.random.rand(sampling_n)
    ##由于q分布随便取，这里就直接用均匀分布了
    weight = np.squeeze(f(w_) / w_)
    weight /= np.sum(weight)

    index = np.random.choice(n*10, n, p=weight)
    sample = candidate[index]
    return sample


def rejection_sampling(X, f, k, n):
    sample = []
    while len(sample) < n:
        candidate = np.random.choice(X)
        ##这里的qsample只能用于一维的计算，没写成矩阵形式
        expec = f(candidate) / (k*qsample(candidate))
        if np.random.rand() < expec:
            sample.append(candidate)
    sample = np.asarray(sample)
    return sample


def qsample(x, mu=0, sigma=1):
    return (1/np.sqrt(2*np.pi*sigma*sigma)) * np.exp(-1*(x - mu)**2/(2*sigma**2))


def metropolis(X, f, n):
    sample = np.zeros(n)
    sample[0] = np.random.choice(X) ##初始的转移值，即可理解为当前状态
    count = 1
    while count < n:
        qs = np.random.choice(X)
        u = np.random.rand()
        alpha = (f(qs) * qsample(sample[count])) / (f(sample[count]) * qsample(qs))
        if u < alpha:#u < min(alpha, 1)改为这个就是Metropolis-Hastings 算法
            sample[count + 1] = qs
            count += 1
        else:
            continue
    return sample

