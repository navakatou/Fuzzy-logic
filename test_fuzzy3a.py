import copy
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
import pso_tofuzzy
from sklearn.metrics import mean_squared_error

# Function to find if the value belongs to a trapezoidal interval
def is_in(val, inter): 
    lim_i = inter[0]
    lim_s = inter[-1]
    if val >=lim_i and val <= lim_s:
        return 1
    else:
        return 0

# Function to find the fuzzy group of a fuzzy set before prediction
def find_grp(seq,x):
    for sublist in seq:
        if sublist[0] == x:
            return seq.index(sublist)

# Function to get a list with the intervals for any element of FLGRn
def get_trp_funs(intvs,grp):
    bsgr = []
    for n in range(1,len(grp)):
        bsgr.append(intvs[grp[n]])
    
    return bsgr

# Function to find duplicate values 
# It returns the all values that are the same and their indexes
def check_dupl(vectr): 
    Not_rep = []
    ind_r = []

    for i, item in enumerate(vectr):
        if item not in Not_rep:
            Not_rep.append(item)
        else:
            ind_r.append(i)

    for i in ind_r:
        a = vectr[i]
        if a in vectr:
            k = vectr.index(a)
            if k not in ind_r:
                ind_r.append(k)
    ind_r = sorted(ind_r)

    return Not_rep, ind_r

# Function to eliminate duplicate values in the list
def red_grp (lsts, inds):
    mod = []
    copa = copy.deepcopy(lsts)
    for item in inds:
        now = copa[item]
        bef = lsts[item-1][0]
        now.insert(0,bef)
        mod.append(now)

    for k in range(len(inds)):
        lsts[inds[k]] = mod[k]

    return lsts
# Function to compute the mean between the difference of consecutive values
def mean_dif (serie):
    sum1 = 0
    vecsz = len(serie)
    for i in range(vecsz-1):
        tmp = np.abs(serie[i+1]-serie[i])
        sum1 = sum1 + tmp
    promd = sum1/(vecsz - 1) 
    return promd

# Function to make the clusters of any length
def do_clusters(vect):

    nvect = sorted(list (set(vect)))
    #print nvect
    svect = len(nvect)
    avgd = mean_dif(nvect) # main average_dif 
    Clstrs = []
    crt_clst = []
    ant_clst = []
    crt_clst.append(nvect[0])
    for k in range(1,svect):
        if (ant_clst == []):
            if (nvect[k] - crt_clst[0]) <= avgd:
                crt_clst.append(nvect[k])
                Clstrs.append(crt_clst)
            else:
                Clstrs.append(copy.deepcopy(crt_clst))
                ant_clst = copy.deepcopy(Clstrs[-1])
                crt_clst[:] = []
                crt_clst.append(nvect[k])
            
        elif (ant_clst != [] and len(crt_clst) == 1):
            dif1 = nvect[k] - crt_clst[0]
            dif2 = crt_clst[0] - ant_clst[-1]
            if (dif1 <= avgd and dif1 < dif2):
                crt_clst.append(nvect[k])
                if Clstrs[-1][0]==crt_clst[0]:
                    Clstrs.pop()
                    Clstrs.append(copy.deepcopy(crt_clst))
                else:
                    Clstrs.append(copy.deepcopy(crt_clst))
                
            else:
                ant_clst = copy.deepcopy(Clstrs[-1])
                Clstrs.append(copy.deepcopy(crt_clst))
                crt_clst[:] = []
                crt_clst.append(nvect[k])

        elif (len(crt_clst)>1):
            clust_dif = mean_dif(crt_clst)
            difa = nvect[k] - crt_clst[-1]
            if (difa <= avgd and difa <= clust_dif):
                crt_clst.append(nvect[k])
                if Clstrs[-1][0]==crt_clst[0]:
                    Clstrs.pop()
                    Clstrs.append(copy.deepcopy(crt_clst))
                else:
                    Clstrs.append(copy.deepcopy(crt_clst))
            else:
                ant_clst = copy.deepcopy(Clstrs[-1])
                crt_clst[:] = []
                crt_clst.append(nvect[k])
                
    return Clstrs , avgd

#Function to reduce the size of the clusters, where the ideal length is two but there could be cases where the length is one
def make_clstrs(clusters, avg_d):
    nclustrs = []
    iclustrs = []
    Clustrs_fin = []

    for item in clusters:
        clssz = len(item)
        tmp = []
        if clssz > 2:
            a = item[0]
            tmp.append(a)
            b = item[-1]
            tmp.append(b)
            nclustrs.append(copy.deepcopy(tmp))
            
        elif clssz == 2:
            nclustrs.append(item)
            
        elif clssz == 1:
            inf = item[0] - avg_d
            tmp.append(inf)
            tmp.append(item[0])
            sup = item[0] + avg_d
            tmp.append(sup)
            smllval = min(clusters[clusters.index(item) -1])
            
            if item == clusters[0]:
                del tmp[0]
                nclustrs.append(copy.deepcopy(tmp))
            elif item == clusters[-1]:
                del tmp[-1]
                nclustrs.append(copy.deepcopy(tmp))
            elif inf < smllval:
                nclustrs.append(copy.deepcopy(item))
            else:
                del tmp[1]
                nclustrs.append(copy.deepcopy(tmp))
                
    return nclustrs

#Function to make the continuos intervals, where the intervals is just between two values. It is based on the algorithm of the paper
def cntns_intvls(grp_cltrs):
    Intrvls = []
    Intrvls.append(grp_cltrs[0])

    for item in grp_cltrs:
        nint = []
        if item != grp_cltrs[0]:
            antr = Intrvls[-1]
            clstsz = len(item)
            if clstsz == 2 :
                dj = antr[-1]
                dk = item[0]
                if dj >= dk:
                    item[0]=dj
                    Intrvls.append(copy.deepcopy(item))
                elif dj < dk:
                    nint.append(dj)
                    nint.append(dk)
                    Intrvls.append(copy.deepcopy(nint))
                    Intrvls.append(copy.deepcopy(item))
            if clstsz == 1 :
                dnk = item[0]
                antr[-1] = dnk
    return Intrvls
                
# Function to increase the intervals and the beginning and final from a set of intervals
def intrvls_aumntd (nintervls, dif_avg, p, q):
    lim_inf = nintervls[0][0]
    lim_sup = nintervls[-1][-1]
    Infs = []
    Sups = []
    
    for i in range(1,p+1):
        tmpp = []
        ninf = lim_inf - 2*i*dif_avg
        tmpp.append(ninf)
        if i == 1:
            nsup = ninf + 2*i*dif_avg
            tmpp.append(nsup)
            intr = ninf
        else:
            nsup = intr
            tmpp.append(nsup)
            intr = ninf
        Infs.append(tmpp)
    Infs = Infs[::-1]
    for i in range(1,q+1):
        nmpp = []
        nsu = lim_sup + 2*i*dif_avg
        if i == 1:
            nin = lim_sup
            nmpp.append(nin)
            nmpp.append(nsu)
            intsr = nsu
        else:
            nin = intsr
            nmpp.append(nin)
            intsr = nsu
            nmpp.append(nsu)
        Sups.append(nmpp)
    
        Final_ints = Infs + nintervls + Sups
    return Final_ints

# Function to creata the matrix of membership, called A, I did not find it usefull
def matrix_A(mat_sh):
    AA = np.zeros((mat_sh,mat_sh))
    b = [0.5, 1.0, 0.5]
    bf = [0.5, 1.0]
    bs = [0.5,1.0]
    m = len(b)
    nc = len(bf)
    for i in range(mat_sh):
        if i == 0:
            AA[i][i:nc]=bf
        elif i == (mat_sh-1):
            AA[i][i+1-nc:i+1]=bs
        else:
            AA[i][i-1:(i+nc)] = b

    return AA

# Function to classify all the FLGR's obteined, check if everyone is upward, equal or downward
def clasf_ftlrg(ftlgrp):
    upwardt = []
    equalt = []
    downwardt = []
    for item in ftlgrp:
        if item[1]<item[0]:
            downwardt.append(item)
        elif item[1]>item[0]:
            upwardt.append(item)
        elif item[1]==item[0]:
            equalt.append(item)

    return upwardt , equalt, downwardt

# Function to get the weights for every tendency (upward, equal and downward) in general
def get_weights(vect1, vobj, nit):
    
    vobj = np.matrix(vobj).T
    Weights = []
    Avg_er = []
    for i in range (nit):
        wt = []
        w1 = np.random.random(1)
        w2 = 1-w1
        wt.append(w2)
        wt.append(w1)
        pnd = np.array(wt)
        error = abs(np.matmul(vect1,pnd) - vobj)
        avg_e = np.mean(error)
        Weights.append(wt) 
        Avg_er.append(avg_e)

    b_avg = Avg_er.index(min(Avg_er))
    b_weits = Weights[b_avg]
    
    return b_weits

#Function to check the tendency of just one FLGR
def eval_tend(valdat):    
    if valdat[1] < valdat[0]:
        tend = 0 # Downward
    elif valdat[1] > valdat[0]:
        tend = 2 # Upward
    elif valdat[1] == valdat[0]:
        tend = 1 # Equal

    return tend

# The programm is based on the paper "Fuzzy Forecasting Based on Fuzzy-Trend Logical
#Relationship Groups". I just worked with the values of TAEIX so the data are the following :

# The set of data training to get the weights
data2 = [4612.97, 4692.17, 4718.81, 4772.61, 4803.51, 4782.82, 4856.82, 4904.93, 4947.16, \
4939.07, 5022.96, 5045.94, 4964.22, 5102.64, 5135.35, 5131.42, 5122.01, 5243.60, 5298.99, \
5341.49, 5312.67, 5390.05, 5391.63, 5237.37, 5217.63, 5204.55, 5072.04, 5061.60, 5088.10, \
5023.41, 5026.43, 4859.48, 4766.91, 4839.42, 4995.32, 4950.37, 5074.80, 5070.24, 4987.73, \
4951.50, 5019.22, 5031.69, 5142.42, 5144.37, 5056.64, 5066.40, 5033.53, 4988.08, 4986.53,\
4923.03, 4926.20, 4965.48, 4984.86, 5030.63, 5029.61, 5012.96, 4890.32, 4851.95, 4810.23,\
4818.18, 4745.03, 4685.16, 4790.10, 4721.25, 4792.54, 4817.35, 4833.47, 4800.94, 4676.48,\
4733.96, 4652.10, 4613.74, 4552.46, 4528.05, 4563.85, 4487.62, 4405.44, 4461.98, 4439.66,\
4538.52, 4503.22, 4527.81, 4547.74, 4556.09, 4505.07, 4470.31, 4480.12, 4441.57, 4486.45,\
4523.95, 4513.19, 4496.19, 4545.18, 4532.53, 4539.47, 4536.53, 4482.25, 4481.29, 4462.61,\
4428.97, 4268.17, 4289.24, 4326.38, 4300.60, 4292.46, 4356.07, 4437.63, 4500.82, 4560.00,\
4563.67, 4623.94, 4637.59, 4634.33, 4540.00, 4541.38, 4501.02, 4519.27, 4496.58, 4456.85,\
4468.94, 4583.97, 4586.61, 4565.04, 4526.65, 4633.80, 4597.19, 4681.07, 4676.13, 4672.49,\
4637.78, 4603.90, 4628.78, 4587.85, 4618.02, 4697.66, 4690.48, 4655.08, 4565.01, 4501.97,\
4496.44, 4467.50, 4523.81, 4454.08, 4455.98, 4489.63, 4511.97, 4432.22, 4329.31, 4337.99,\
4316.79, 4384.52, 4312.38, 4307.69, 4352.51, 4349.68, 4282.60, 4248.55, 4147.59, 4159.02,\
4154.80, 3978.87, 4022.16, 4018.85, 4008.22, 3959.10, 3947.32, 4029.53, 4108.52, 4101.84,\
4072.08, 4088.18, 4134.91, 4122.78, 4082.90, 4030.73, 3951.73, 3958.28, 3901.61, 3893.34,\
3914.90, 3926.38, 3784.48, 3772.91, 3733.93, 3733.06, 3866.22, 3822.81, 3850.68, 3816.38,\
3790.03, 3800.97, 3852.45, 3946.35, 3960.45, 3933.61, 3888.49, 3852.27, 3758.38, 3770.12,\
3785.50, 3685.04, 3716.72, 3664.52, 3441.69, 3437.60, 3569.93, 3481.90, 3524.75, 3529.42,\
3540.48, 3472.50, 3351.63, 3397.29, 3524.21, 3584.08, 3717.71, 3704.25, 3720.45, 3562.87,
3592.12, 3688.21, 3622.75, 3602.83, 3602.96, 3633.22, 3729.02, 3707.18, 3749.32, 3734.51,\
3724.17, 3694.38, 3699.23, 3662.60, 3670.90, 3647.97, 3670.90, 3640.90, 3631.73]

# Set of data test to be used in the forecasting part
data2_af = [3559.51, 3559.66, 3506.05, 3593.41, 3532.61, 3535.78, 3530.14, 3534.53, 3570.89,\
3555.51, 3573.09, 3588.44, 3583.71, 3660.10, 3682.38, 3724.33, 3711.71, 3698.51, 3687.22, 3673.51, \
3676.56, 3681.94, 3686.14, 3675.01, 3646.76, 3635.70, 3614.08, 3651.39, 3727.95, 3755.77, 3761.01, \
3776.55, 3746.75, 3734.30, 3742.61, 3696.76, 3668.67, 3657.99, 3576.09, 3579.97, 3448.15, 3456.00, \
3327.67, 3377.06]

# The dates of the data test
dates_af = ['11/2/92', '11/3/92', '/11/4/92', '11/5/92', '11/6/92', '11/7/92', '11/9/92',\
'11/10/92', '11/11/92', '11/13/92', '11/14/92', '11/16/92', '11/17/92', '11/18/92', '11/19/92',\
'11/20/92', '11/21/92', '11/23/92', '11/24/92', '11/25/92', '11/26/92', '11/27/92', '11/28/92', \
'11/30/92', '12/1/92', '12/2/92', '12/3/92', '12/4/92', '12/5/92', '12/7/92', '12/8/92', '12/9/92',\
'12/10/92', '12/11/92', '12/12/92', '12/14/92', '12/15/92', '12/16/92', '12/17/92', '12/18/92', \
'12/21/92', '12/22/92', '12/23/92', '12/24/92', '12/28/92', '12/29/92']

Clusters, avd = do_clusters(data2)
siz_dat = len(data2)
print ''
print 'Size of data {}'.format(len(data2))
nClusters = make_clstrs(Clusters, avd)
intervls = cntns_intvls(nClusters)
print 'Size of intervals not aumented {}'.format(len(intervls))
n_intervls = intrvls_aumntd(intervls, avd, 40, 13)
szint = len(n_intervls)
print 'Size final intervals {}'.format(szint)
matr_A = matrix_A(szint)
A_names = [i for i in np.arange(szint)] 

# Fuzzyfy the data set using the trapezoid fuzzyfication approach
# It's necessary to delete duplicate values fuzzified. TODO ??

training_d = data2[:] 
Fzytion_int = []    
Fzytion_set = []
Fzytion_val = []
for val in training_d:
    for p in range(szint):
        xset = is_in(val,n_intervls[p])
        if xset == 1:
            tmpi = A_names[p]
            Fzytion_set.append(int(A_names[p]))
            Fzytion_val.append(val)
            Fzytion_int.append(n_intervls[p])
            break # This takes the first interval where the value is in

print ''
colnames = ['TAEIX', 'Fuzzset', 'Interval']
Fuzzif_dat = pd.DataFrame({'TAEIX':Fzytion_val,'Fuzzset':Fzytion_set, 'Interval':Fzytion_int}, columns = colnames)
Fuzzif_dat.to_csv('TAEIX_fuzz.csv', sep=',')

# Create a list of first order relationships First_or = [[Ai,Aj], [Aj,Ak], ....]
n = 3 # Number of order 
cs = 1 # For 1st order
fzzy_grp = list (Fuzzif_dat['Fuzzset'][k:k+n].values.tolist() for k in range (0, len(Fuzzif_dat['Fuzzset'])-(n-1), cs))
print 'Number of FSGs {}'.format(len(fzzy_grp))

First_or, indx = check_dupl(fzzy_grp)
band = True
cont = 0
while ( band and cont < 5):
    fzzy_grp = red_grp(fzzy_grp, indx)
    First_or, indx = check_dupl(fzzy_grp)
    if (indx == []):
        band = False
        print 'Indexes is empy'
    cont += 1

print 'Fuzzy set groups'
print len(fzzy_grp)

Up, Eq, Down = clasf_ftlrg(fzzy_grp)
Up = np.array(Up)
Eq = np.array(Eq)
Down = np.array(Down)

train_up = Up[:,0:2]
train_eq = Eq[:,0:2]
train_down = Down[:,0:2]
targ_up = Up[:,2]
targ_eq = Eq[:,2]
targ_down = Down[:,2]

we_up = get_weights(train_up, targ_up, 300)
we_eq = get_weights(train_eq, targ_eq, 300)
we_down = get_weights(train_down, targ_down, 300)
Weights_f = [we_down, we_eq, we_up]
print 'Final weigths {}'. format(Weights_f)

# Process of Defuzzyfication, based on the paper, at the end is just to find a fuzzy set and take the midpoint.
# The beginning is just take the last two elements of the training set and star the computation
Actl_vals = []
Forcst_vals = []
Actl_vals.append(data2[siz_dat-2])
Actl_vals.append(data2[siz_dat-1])

for i in range(len(data2_af)):
    vpast = Actl_vals[i:i+2]
    pstfz = []
    for valor in vpast:
        for p in range(szint):
            xset = is_in(valor,n_intervls[p])
            if xset == 1:
                seti = A_names[p]
                pstfz.append(int(A_names[p]))
                break # This takes the first interval where the value is in
    vltnd = eval_tend(pstfz)
    indfz = np.ceil(np.matmul(np.array(Weights_f[vltnd]).T,np.array(pstfz)))
    forcs_va = np.mean(n_intervls[int(indfz)])
    Forcst_vals.append(forcs_va)
    Actl_vals.append(data2_af[i])

rmse = np.sqrt(mean_squared_error(data2_af, Forcst_vals))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.figure()
plt.plot(data2_af, marker = '^', linestyle = '-', color = 'r', label = 'Original')
plt.plot(Forcst_vals, marker = '*', linestyle = '--', color = 'b' , label = 'Forecasted')
plt.figtext(0.75, 0.3, 'RMSE = {}'.format(rmse), fontdict =  None,
            horizontalalignment = 'center', fontsize = 12, bbox = props)
plt.legend(loc='upper left')
plt.xlabel('Dates')
plt.ylabel('Production')
plt.title('Forecasted values of TAIEX ')
plt.xticks(range(len(dates_af)), dates_af)
plt.xticks(range(len(dates_af)), dates_af, rotation=45) #writes strings with 45 degree angle
plt.grid()
plt.show()
