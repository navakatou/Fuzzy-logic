# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:40:29 2017

@author: albertnava
"""
# %%

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import itertools 

# %%
# Membership functions for fuzzy logic (except Gaussian)

# Function to get the degree of a value x in a incremental ramp fuzzy interval
def up(a, b, x):
    a = float(a)
    b = float(b)
    x = float(x)
    if x < a:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return 1.0

# Function to get the degree of a value x in a decremental ramp fuzzy interval
def down(a, b, x):
    return 1. - up(a, b, x)

# Function to get the degree of a value x in a triangular fuzzy interval
def tri(a, b, x):
    a = float(a)
    b = float(b)
    m = (a + b) / 2.
    first = (x - a) / (m - a)
    second = (b - x) / (b - m)
    return max(min(first, second), 0.)

# Function to get the degree of a value x in the trapezoidal fuzzy interval
def trap(A, x):
    
    try:
        if len(A)==4 :
            a = float(A[0])
            b = float(A[1])
            c = float(A[2])
            d = float(A[3])    
            
            m1 = 1.0/(b-a)
            b1 = -a/(b-a)
            m2 = -1.0/(d-c)
            b2 = d/(d-c)
            if a<= x <= b :
                val = float((m1*x + b1))
                #print 'First ramp'
            if b<= x <= c :
                val = float(1)
                #print 'Flat'
            if c<= x <= d :
                val = float((m2*x+b2))
                #print 'Second ramp'
            if (x<a or x>d):
                val = 0.0
                print 'Aqui caigo', A, x
            #print 'Trapezoidal degree', x
            return val
    except:
        
        print 'Vector must have 4 elements', x, A
    
    #first = (x - a) / (b - a)
    #second = (d - x) / (d - c)
    
    

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
    #        yind = seq.index(sublist)
    #    if sublist[0] != x:
    #        yind = -1
    #return yind

# Function to find the mid-point of the group of trapezoidal fuzzy intervals
def defuz_midpoint(fgrl):
    t_v = len(fgrl)
    if t_v > 1:
        midp = 0
        for item in fgrl:
            tmp = np.mean([item[1],item[2]])
            midp += tmp
        midp = midp/t_v
    else:
        midp = np.mean([fgrl[0][1],fgrl[0][2]])
        
    return int(midp)

# Function to get a list with the intervals for any element of FLGRn
def get_trp_funs(intvs,grp):
    bsgr = []
    for n in range(1,len(grp)):
        bsgr.append(intvs[grp[n]])
    
    return bsgr

# Function to get the minimum degree of duplicate fuzzyfied elements (element, Ai)
def get_max_mf(gropued):
    tamDF = len(gropued)
    degmf = []
    #print 'Gruops of duplicates', gropued
    for i in range(tamDF):
        temtr = trap(gropued.iloc[i][2],gropued.iloc[i][1])
        degmf.append(temtr)
    #valmax = np.max(degmf)
    valmax = np.min(degmf)    
    indxt = degmf.index(valmax)
    max_ind = gropued.iloc[indxt][0]

    return max_ind
    
# %% I decided to work with the data here, maybe is necessary to add a function
#    to read any document with the help of pandas

# Information about arrivals to New Zeland through the years 2000-2013 
dates1 = ['Jan-2000','Feb-2000', 'Mar-2000', 'Apr-2000', 'May-2000', 'Jun-2000',\
'Jul-2000', 'Aug-2000', 'Sep-2000', 'Oct-2000', 'Nov-2000', 'Dec-2000', 'Jan-2001', \
'Feb-2001', 'Mar-2001', 'Apr-2001', 'May-2001', 'Jun-2001', 'Jul-2001', 'Aug-2001',\
'Sep-2001', 'Oct-2001', 'Nov-2001', 'Dec-2001', 'Jan-2002', 'Feb-2002', 'Mar-2002', \
'Apr-2002', 'May-2002', 'Jun-2002', 'Jul-2002', 'Aug-2002', 'Sep-2002', 'Oct-2002',\
'Nov-2002', 'Dec-2002', 'Jan-2003', 'Feb-2003', 'Mar-2003',  'Apr-2003', 'May-2003', \
'Jun-2003', 'Jul-2003', 'Aug-2003', 'Sep-2003', 'Oct-2003', 'Nov-2003', 'Dec-2003', \
'Jan-2004', 'Feb-2004', 'Mar-2004', 'Apr-2004', 'May-2004', 'Jun-2004', 'Jul-2004', \
'Aug-2004', 'Jan-2004', 'Feb-2004', 'Mar-2004', 'Apr-2004', 'May-2004', 'Jun-2004',\
'Jul-2004', 'Aug-2004', 'Sep-2004', 'Oct-2004', 'Nov-2004', 'Dec-2004', 'Jan-2005',\
'Feb-2005', 'Mar-2005', 'Apr-2005', 'May-2005', 'Jun-2005', 'Jul-2005', 'Aug-2005',\
'Sep-2005', 'Oct-2005', 'Nov-2005', 'Dec-2005', 'Jan-2006', 'Feb-2006', 'Mar-2006',\
'Apr-2006', 'May-2006', 'Jun-2006', 'Jul-2006', 'Aug-2006', 'Sep-2006', 'Oct-2006',\
'Nov-2006', 'Dec-2006', 'Jan-2007', 'Feb-2007', 'Mar-2007', 'Apr-2007', 'May-2007',\
'Jun-2007', 'Jul-2007', 'Aug-2007', 'Sep-2007', 'Oct-2007', 'Nov-2007', 'Dec-2007',\
'Jan-2008', 'Feb-2008', 'Mar-2008', 'Apr-2008', 'May-2008', 'Jun-2008', 'Jul-2008',\
'Aug-2008', 'Sep-2008', 'Oct-2008', 'Nov-2008', 'Dec-2008', 'Jan-2009', 'Feb-2009',\
'Mar-2009', 'Apr-2009', 'May-2009', 'Jun-2009', 'Jul-2009', 'Aug-2009', 'Sep-2009',\
'Oct-2009', 'Nov-2009', 'Dec-2009', 'Jan-2010', 'Feb-2010', 'Mar-2010', 'Apr-2010',\
'May-2010', 'Jun-2010', 'Jul-2010', 'Aug-2010', 'Sep-2010', 'Oct-2010', 'Nov-2010',\
'Dec-2010', 'Jan-2011', 'Feb-2011', 'Mar-2011', 'Apr-2011', 'May-2011', 'Jun-2011',\
'Jul-2011', 'Aug-2011', 'Sep-2011', 'Oct-2011', 'Nov-2011', 'Dec-2011', 'Jan-2012',\
'Feb-2012', 'Mar-2012', 'Apr-2012', 'May-2012', 'Jun-2012', 'Jul-2012', 'Aug-2012',\
'Sep-2012', 'Oct-2012', 'Nov-2012', 'Dec-2012', 'Jan-2013', 'Feb-2013', 'Mar-2013']

students1 = [169404, 192856, 152910, 143681, 99068, 97516, 130571, 117365, 113750, 146610,\
182324, 243023, 197765, 199792, 176875, 153186, 110936, 112279, 144380, 136864, 131194,\
142095, 164636, 239807, 204717, 212233, 202504, 143877, 118201, 115194, 152156, 133272,\
136085, 162327, 198705, 265691, 220861, 222201, 193853, 150416, 102745, 111982, 145564,\
135351, 148420, 165821, 211735, 297280, 244333, 238032, 211748, 184379, 132715, 134813,\
173328, 152104, 244333, 238032, 211748, 184379, 132715, 134813, 173328, 152104, 161182,\
181371, 220610, 313057, 249933, 250070, 234101, 174757, 135708, 157547, 168422, 150656,\
163785, 176216, 214694, 307061, 250554, 252431, 226966, 191648, 135279, 139891, 166970,\
155699, 166531, 186639, 229913, 319040, 246748, 267569, 239203, 193229, 140755, 145498,\
173046, 164775, 168838, 179947, 228813, 317259, 253515, 280513, 250806, 179388, 140483,\
142413, 175738, 162485, 157704, 173938, 219313, 322207, 244030, 256559, 226461, 195883,\
141916, 135162, 176198, 161100, 172425, 187372, 219939, 341337, 256652, 267855, 243263,\
187962, 141336, 145825, 182904, 168081, 174157, 184898, 226455, 345656, 265553, 268259,\
215553, 197777, 140741, 131269, 176084, 175909, 219940, 215902, 230292, 364165, 266839,\
259083, 239929, 195668, 140841, 151074, 173539, 178298, 179069, 184200, 232119, 363959,\
260637, 281233, 270740]

dates = [1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985,\
1986, 1987, 1988, 1989, 1990, 1991, 1992]
students = [13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807, 16919, 16388, 15433, \
15497, 15145, 15163, 15984, 16859, 18150, 18970, 19328, 19337, 18876]

test_d = pd.Series(students)


Df= pd.DataFrame({ 'Enrollment': students, 'Year' : dates }, columns = ['Year', 'Enrollment'] )    

# Sorting the data in ascending order
new_stud = np.sort(students)
Dmax = max(students)
Dmin = min(students)
print 'Maximum data value', Dmax 
print 'Minimum data value', Dmin

l_tam = len(dates)
# Computing average distance between data sorted
dist = []
dist2 = []
sum1 = 0
sum1a = 0
for i in range(l_tam-1):
    tmp = np.abs(new_stud[i]-new_stud[i+1])
    tmp1 = np.abs(students[i+1]-students[i])
    sum1a = sum1a + tmp1
    sum1 = sum1 + tmp
    dist2.append(tmp1)
    dist.append(tmp)
AD = sum1/(l_tam - 1)
Range = sum1a/(2*(l_tam-1))
print 'Average Distance 1', Range
print 'Average Distance', AD

# Computing standard deviation
sum2 = 0
for j in range (l_tam-1):
    tm = (dist[j]-AD)**2
    sum2 = sum2 + tm
sAD = np.sqrt(sum2 / l_tam)
print sAD

best_dist = []
#sum3 = 0
for k in range(l_tam-1):
    if (dist[k]>= (AD-sAD) and dist[k]<= (AD+sAD)):
        #sum3 = sum3 + dist[k]
        best_dist.append(dist[k])
        #print dist[k]
ADr = round(np.mean(best_dist),0)
print 'New average distance', ADr
# Limits of universe U = [Dmin-AD,Dmax+AD]
l_inf = Dmin - ADr
l_sup = Dmax + ADr
print 'New inferior limit', l_inf
print 'New superior limit', l_sup
rang = np.arange(l_inf,l_sup)
# Number of sets of U, n = (R-S)/2S 
R = l_sup - l_inf
S = 190  
n = round((R-S)/(2*S),0)
A_names = [str(i) for i in np.arange(n)] # There is no error of np.arange
n_p = (2*n)


print 'Number of intervals 1 =', n 
m = R/Range
print 'Number of intervals 2= ', m

nl_inf = l_inf + ADr
nl_sup = l_sup - ADr

#Create the intervals of U
U_s = []
U_l = []
U_t = []
for i in range(1,int(n_p)):
    U_l.append(nl_inf+(i-1)*S)  
    if len(U_l) == (int(n_p)-1):
        U_l.append(nl_inf+i*S)  
        #print "appended"
    U_s.append(nl_inf+i*S)
    #U_t.append((nl_inf+i*S)-(nl_inf+(i-1)*S))

#print U_l
#print len(U_l)

U_l.append(l_sup)
U_l.insert(0,l_inf)

# Create a list of lists with parameters to form trapezoidal MF
Anew = []
x = []
for i in range(len(U_l)):
    x.append(U_l[i])
    if len(x)==4:
        tmp = x[:]
        Anew.append(tmp)
        del x[0]
        del x[0]
#print Anew    
#print len(Anew)
A = [[12861,13055,13245,13436], [13245,13436,13626,13816],[13626,13816,14007,14197], \
[14007,14197,14388,14578], [14388,14578,14768,14959], [14768,14959,15149,15339], \
[15149,15339,15530,15720], [15530,15720,15910,16101], [15910,16101,16291,16482], \
[16291,16482,16672,16862], [16672,16862,17053,17243], [17053,17243,17433,17624], \
[17433,17624,17814,18004], [17814,18004,18195,18385], [18195,18385,18576,18766], \
[18576,18766,18956,19147], [18956,19147,19337,19531]]

# Create all trapezoidal membershipfunctions
#fuzz_set_A = ctrl.Consequent(rang,'universe')
#for k in range(int(n)) :
#    fuzz_set_A[A_names[k]] = fuzz.trapmf(fuzz_set_A.universe,Anew[k])
#fuzz_set_A.view()
    
# Fuzzyfy the data set using the trapezoid fuzzyfication approach
Fzytion_num = []    
Fzytion_set = []
Fzytion_val = []
for val in students:
    for p in range(len(Anew)):
        #xset = is_in(val,A[p])
        xset = is_in(val,Anew[p])
        if xset == 1:
            Fzytion_set.append(int(float(A_names[p]))) # Assigment of fuzzy set Ai
            Fzytion_val.append(val) # Values of the fuzzy set Ai
            Fzytion_num.append(Anew[p])
            #Fzytion_num.append(A[p])
        
vc = pd.Series(Fzytion_val)
df =vc[vc.duplicated(keep=False)]
dfp  = df.tolist() # List of duplicate values
inx_d = df.index.tolist() # List of indexes of duplicate values
#print (Fzytion_val)                    
#print (Fzytion_set)
#print (dfp)
#print (inx_d)
print ''

# Elimiar uno de los valores que se repitan, al considerar el que tiene el mayor grado en
# su funcion mu

itx = list(range(len(Fzytion_val))) # Indexes just to check wich rows will be keep
col_names = ['indc', 'estd', 'fuzznum', 'Aip']
setg = pd.DataFrame({'estd':Fzytion_val,'indc':itx, 'fuzznum':Fzytion_num, 'Aip':Fzytion_set}, columns = col_names)
#print setg.head(7)
g = setg.groupby(['estd']).filter(lambda x : len(x)>1)
g = g.groupby('estd') # Just to get the groups with duplicate values


idx_d = [] # Indexes wich have higher degree of membership function
for name,group in g:
    fizt = get_max_mf(group)
    #print  group
    idx_d.append(fizt)
#print idx_d

nidx = [item for item in itx if item not in idx_d] 
Nsetg = setg[setg.index.isin(nidx)].copy(deep=True) # Final Data frame with unique elements of students (is not the same
# as the values of the paper because of the partition is a little bit different)
#print len(Nsetg)
#print 'New index'
#print nidx

# %%
# Create a list of first order relationships First_or = [[Ai,Aj], [Aj,Ak], ....]
n = 2 # Number of order 
cs = 1 # For 1st order
st_or = list (Nsetg['Aip'][k:k+n].values.tolist() for k in range (0, len(Nsetg['Aip'])-1, cs))

#for j,a in enumerate(st_or):
#    print j,a 
st_or.sort()
First_or = list(k for k,_ in itertools.groupby(st_or))
#print First_or
#print len(First_or)
# Create the list of the First Group Relationships FLGRn
FLGRs = [] # List of groups (indexes) with the same initial set
ukeys = [] # List of initial set (index)
for k,g in itertools.groupby(First_or, lambda x : x[0]):
    #print list(g)    
    FLGRs.append(list(g))
    ukeys.append(k)

FLGRn = [] 
for i,e in zip(ukeys,FLGRs):
    tmp = []
    tmp.append(i)
    #print (i,e)
    for it in e:
        tmp.append(it[1])
    FLGRn.append(tmp)

#print FLGRn



# %%
# The process of "Deffuzyfication" by calculating the centroid 

ai_g = [] # Just a vector to keep the indexes of FLGRn for each Ai
aine = [0] # Vector for values predicted
for item in Nsetg['Aip']:
    gii = find_grp(FLGRn,item)
    if gii != None:
        gind = FLGRn[gii]
        ai_g.append(gii)
        intervs = get_trp_funs(Anew,gind)
    else:
        gind = [Anew[item]]
        ai_g.append(item)
        print 'Value whitout group',item, gind
        intervs = gind
    #print 'Calculate midpoint for value', item, gind
    #intervs = get_trp_funs(Anew,gind)
    midpoint = defuz_midpoint(intervs)
    aine.append(midpoint)
print ''
print aine

# %% 

#tt = Anew[9]
#x_T = 
