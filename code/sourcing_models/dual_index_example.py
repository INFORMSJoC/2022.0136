import numpy as np
import matplotlib.pyplot as plt
from lib.dual_sourcing import *

np.random.seed(10)

ce = 20
cr = 0
le = 0
lr = 2
ze = 100
h = 5
b = 495
T = 200

samples = 1000
Delta_arr = np.arange(0,7)

optimal_ze, optimal_Delta = dual_index_ze_Delta(samples,
                                                Delta_arr,
                                                ce, 
                                                cr, 
                                                le, 
                                                lr,
                                                h, 
                                                b, 
                                                T,
                                                ze)

T = 200000
S = DualSourcingModel(ce=ce, 
                      cr=cr, 
                      le=le, 
                      lr=lr, 
                      h=h, 
                      b=b,
                      T=T, 
                      I0=optimal_ze,
                      ze=optimal_ze,
                      Delta=optimal_Delta,
                      dual_index=True)

S.simulate()  

print("average cost (dual index):", S.total_cost/T)

plt.figure()
plt.plot(S.inventory, '-o', label = r"inventory")
plt.plot(S.demand, '-o', label = r"demand")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(S.qe, '-o', label = r"expedited order")
plt.plot(S.qr, '-o', label = r"regular order")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(S.cost, '-o', label = r"cost")
plt.xlabel(r"time")
plt.ylabel(r"cost")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()
