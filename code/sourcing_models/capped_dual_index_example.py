import numpy as np
import matplotlib.pyplot as plt
from lib.dual_sourcing import *

#np.random.seed(10)

ce = 10
cr = 0
le = 0
lr = 4
h = 5
b = 495

# T = 30000

# u1_arr = np.arange(1,5)
# u2_arr = np.arange(8,14)
# u3_arr = np.arange(5)

# optimal_u1, optimal_u2, optimal_u3 = capped_dual_index_parameters(u1_arr,
#                                                                    u2_arr,
#                                                                    u3_arr,
#                                                                    ce, 
#                                                                    cr, 
#                                                                    le, 
#                                                                    lr,
#                                                                    h, 
#                                                                    b, 
#                                                                    T)

T = 50
S = DualSourcingModel(ce=ce, 
                      cr=cr, 
                      le=le, 
                      lr=lr, 
                      h=h, 
                      b=b,
                      T=T, 
                      I0=4,
                      u1=4,
                      u2=12,
                      u3=2,
                      capped_dual_index=True)

S.simulate()  

# plt.figure()
# plt.plot([np.mean(S.cost[:i]) for i in range(1,len(S.cost),100000)])
# plt.hlines(np.mean(S.cost),0,T/100000)
# plt.ylim(23.18,23.26)
# plt.xlabel(r"time [in 100000 periods]")
# plt.ylabel(r"average cost")
# plt.tight_layout()
# plt.show()

print("average cost (capped dual index):", S.total_cost/T)

# plt.figure()
# plt.plot(S.inventory, '-o', label = r"inventory")
# plt.plot(S.inventory_position, '-o', label = r"inventory position")
# plt.plot(S.demand, '-o', label = r"demand")
# plt.xlabel(r"time")
# plt.ylabel(r"value")
# plt.legend(loc = 4, ncol = 3)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.plot(S.qe, '-o', label = r"expedited order")
# plt.plot(S.qr, '-o', label = r"regular order")
# plt.xlabel(r"time")
# plt.ylabel(r"value")
# plt.legend(loc = 4, ncol = 3)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.plot(S.cost, '-o', label = r"cost")
# plt.xlabel(r"time")
# plt.ylabel(r"cost")
# plt.legend(loc = 4, ncol = 3)
# plt.tight_layout()
# plt.show()
