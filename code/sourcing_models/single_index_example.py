import numpy as np
import matplotlib.pyplot as plt
from lib.dual_sourcing import *

np.random.seed(10)

ce = 20
cr = 0
le = 0
lr = 2
zr = 100
h = 5
b = 495
T = 200

samples = 1000
Delta_arr = np.arange(0,5)

optimal_zr, optimal_Delta = single_index_zr_Delta(samples,
                                                  Delta_arr,
                                                  ce, 
                                                  cr, 
                                                  le, 
                                                  lr,
                                                  h, 
                                                  b, 
                                                  T,
                                                  zr)

T = 100
S = DualSourcingModel(ce=ce, 
                      cr=cr, 
                      le=le, 
                      lr=lr, 
                      h=h, 
                      b=b,
                      T=T, 
                      I0=optimal_zr,
                      zr=optimal_zr,
                      Delta=optimal_Delta,
                      single_index=True)

S.simulate()  

print("average cost (single index):", S.total_cost/T)

plt.figure()
plt.plot(S.inventory, '-o', label = r"inventory")
plt.plot(S.inventory_position, '-o', label = r"inventory position")
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
