import numpy as np
from tqdm import trange
import torch
# https://arxiv.org/pdf/2306.12422.pdf

def dreamtime_t(m1,m2,s1,s2,min_t,max_t,N):
    wt = compute_wt(m1,m2,s1,s2,min_t,max_t)
    pt = compute_normalized_pt(wt)
    ti = compute_ti(pt,N)
    return ti

def compute_wt(m1,m2,s1,s2,min_t,max_t):
    t_len = max_t-min_t+1
    wt = np.zeros(t_len)
    for i in range(t_len):
        t = i+min_t#防止min_t != 0，一般情况i=t
        if t > m1:
            tmp = np.exp(-(t-m1)**2 / (2 * s1**2))
        if t>=m2 and t<=m1:
            tmp = 1
        if t<m2:
            tmp = np.exp(-(t-m2)**2 / (2 * s2**2))
        wt[i] = tmp
    return wt

def compute_normalized_pt(wt):
    sum_wt = np.sum(wt)
    pt = wt/sum_wt
    return pt

def compute_ti(pt,N):
    ti = np.zeros(N)
    for i in trange(N):
        ti[i] = compute_t_star(pt,N,i)
    return ti

def compute_t_star(pt,N,i):
    tmp_t = 0
    min_value = np.abs(np.sum(pt[tmp_t:]) - float(i+1) / float(N))
    t_star = 0
    for tmp_t in range(1,len(pt)):
        tmp_value = np.abs(np.sum(pt[tmp_t:]) - float(i+1) / float(N))
        if tmp_value < min_value:
            t_star = tmp_t
            min_value = tmp_value
    return t_star+1     