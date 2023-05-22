import numpy as np

class DualSourcingModel:
    def __init__(self, 
                 ce=0, 
                 cr=0, 
                 le=0, 
                 lr=0,
                 h=0, 
                 b=0, 
                 T=200,
                 I0=0,
                 zr=0,
                 ze=0,
                 Delta=0,
                 Q=0,
                 s=0,
                 u1=0,
                 u2=0,
                 u3=0,
                 single_index=False,
                 dual_index=False,
                 tailored_base_surge=False,
                 capped_dual_index=False,
                 demand_distribution=[-1]):
        """ 
        Initialization of dual sourcing model. 
        
        Parameters: 
        ce (int): per unit cost of expedited supply
        cr (int): per unit cost of regular supply 
        le (int): expedited supply lead time
        lr (int): regular supply lead time
        h (int): holding cost per unit
        b (int): shortage cost per unit 
        T (int): number of periods
        I0 (int): initial inventory level
        Delta (int): difference between expedited and 
        regular target order level (i.e., zr-ze) 
        zr (int): regular target order level (single index)
        ze (int): expedited target order level (dual index)
        Q (int): regular order quantity (tailored base surge)
        s (int): underlying single base-stock level (tailored base surge)
        u1 (int): capped dual index parameter 1
        u2 (int): capped dual index parameter 2
        u3 (int): capped dual index parameter 3
        single_index (bool): single index yes/no
        dual_index (bool): dual index yes/no
        tailored_base_surge (bool): tailored base surge yes/no
        capped_dual_index (bool): capped dual index yes/no
        demand_distribution (array): demand distribution that is given
        by an array ([-1] indicates that the standard, uniform 
        distribution should be used)
      
        """

        # lead time and cost conditions
        assert le < lr, "le must be smaller than lr"
        assert ce > cr, "ce must be larger than cr"
        
        self.cost_e = ce
        self.cost_r = cr
        self.lead_time_e = le
        self.lead_time_r = lr
        self.holding_cost = h
        self.shortage_cost = b     
        
        self.current_demand = 0
        self.current_inventory = I0
        self.current_inventory_position = self.current_inventory
        self.current_cost = 0

        # current order quantities
        self.current_qe = 0
        self.current_qr = 0
        
        if self.lead_time_e == 0:
            self.qe = [self.current_qe]
        else:
            self.qe = self.lead_time_e*[self.current_qe]
        
        if self.lead_time_r == 0:
            self.qr = [self.current_qr]
        else:
            self.qr = self.lead_time_r*[self.current_qr]
            
        # simulation period and containers
        self.period = T
        self.inventory = [self.current_inventory]
        self.inventory_position = [self.current_inventory_position]

        self.cost = [self.current_cost]
        self.demand = [self.current_demand]
        self.total_cost = 0
        
        self.demand_flag = -1
        self.demand_support = [0,1,2,3,4]#,5,6,7,8]
        self.demand_generator = lambda: np.random.choice(self.demand_support)

        if demand_distribution[0] != -1:
            self.demand_flag = 1
            self.demand_generator = lambda i: int(demand_distribution[0](i,1))
        
        # initialize single index policy parameters
        self.single_index = single_index
        if self.single_index:
            self.initialize_single_index(Delta,zr)
        
        # initialize dual index policy parameters
        self.dual_index = dual_index
        if self.dual_index:
            self.initialize_dual_index(Delta,ze)
            
        # initialize tailored base surge parameters
        self.tailored_base_surge = tailored_base_surge
        if self.tailored_base_surge:
            self.initialize_tailored_base_surge(Q,s)
        
        # initialize capped dual index
        self.capped_dual_index = capped_dual_index
        if self.capped_dual_index:
            self.initialize_capped_dual_index(u1,u2,u3)
            
    def initialize_single_index(self,
                                Delta,
                                zr):
        """ 
        Initialization of single index policy parameters. 
        (for more information, see Scheller-Wolf, A., Veeraraghavan, S., 
        & van Houtum, G. J. (2007). Effective dual sourcing with a single 
        index policy. Working Paper, Tepper School of Business, 
        Carnegie Mellon University, Pittsburgh.)
        
        Parameters: 
        Delta (int): difference between expedited and 
        regular target order level (i.e., zr-ze) 
        zr (int): regular target order level
      
        """
        
        self.critical_fractile = self.shortage_cost/ \
        (self.holding_cost+self.shortage_cost)
        
        self.Delta = Delta
        self.target_order_level_r = zr
        self.target_order_level_e = self.target_order_level_r-self.Delta
        
        self.current_inventory_position = self.current_inventory
        self.inventory_position = [self.current_inventory_position]
            
        # measure of the difference between zr and inventory level (Eq. 6 in 
        # Scheller-Wolf, A., Veeraraghavan, S., & van Houtum, G. J. (2007). 
        # Effective dual sourcing with a single index policy. 
        # Working Paper, Tepper School of Business, 
        # Carnegie Mellon University, Pittsburgh.)
        self.single_index_D_Delta = 0
        
    def initialize_dual_index(self,
                              Delta,
                              ze):
        """ 
        Initialization of single index policy parameters. 
        (for more information, see Veeraraghavan, S., & Scheller-Wolf, A. 
        (2008). Now or later: A simple policy for effective dual sourcing 
        in capacitated systems. Operations Research, 56(4), 850-864.)
        
        Parameters: 
        Delta (int): difference between expedited and 
        regular target order level (i.e., zr-ze) 
        zr (int): regular target order level
      
        """
        
        self.lead_time_difference = self.lead_time_r - self.lead_time_e

        self.critical_fractile = self.shortage_cost/ \
        (self.holding_cost+self.shortage_cost)
        
        self.Delta = Delta
        self.target_order_level_r = ze+self.Delta
        self.target_order_level_e = ze
        
        self.current_inventory_position_e = self.current_inventory
        self.current_inventory_position_r = self.current_inventory

        self.inventory_position_e = [self.current_inventory_position_e]
        self.inventory_position_r = [self.current_inventory_position_r]
            
        # measure of the difference between zr and inventory level (Eq. 6 in 
        # Scheller-Wolf, A., Veeraraghavan, S., & van Houtum, G. J. (2007). 
        # Effective dual sourcing with a single index policy. 
        # Working Paper, Tepper School of Business, 
        # Carnegie Mellon University, Pittsburgh.)
        self.dual_index_G_Delta = 0
    
    def initialize_tailored_base_surge(self,
                                       Q,
                                       s):
        """ 
        Initialization of tailored base-surge policy. 
        (for more information, see Allon, G., & Van Mieghem, J. A. (2010). 
        Global dual sourcing: Tailored base-surge allocation to near- 
        and offshore production. Management Science, 56(1), 110-124. and
        Chen, B., & Shi, C. (2019). Tailored base-surge policies in 
        dual-sourcing inventory systems with demand learning. 
        Available at SSRN 3456834.)
        
        Parameters: 
        Q (int): regular order quantity
        s (int): underlying single base-stock level
      
        """
        
        self.regular_order_Q = Q
        self.single_base_stock_level = s
    
    def initialize_capped_dual_index(self,
                                     u1,
                                     u2,
                                     u3):
        """ 
        Initialization of capped dual index policy. 
        (for more information, see Sun, J., & Van Mieghem, J. A. (2019). 
        Robust dual sourcing inventory management: Optimality of capped dual 
        index policies and smoothing. Manufacturing & Service 
        Operations Management, 21(4), 912-931.)
        
        u1 (int): capped dual index parameter 1
        u2 (int): capped dual index parameter 2
        u3 (int): capped dual index parameter 3
        
        """
        
        self.lead_time_difference = self.lead_time_r - self.lead_time_e

        self.s_e_optimal = u1
        
        self.s_r_optimal = u2

        self.q_r_ast = u3
    
    def inventory_evolution(self):
        """ 
        Update inventory position, inventory, and cost.
        """
        
        if self.single_index:
            self.current_inventory_position -= self.current_demand
            self.current_inventory_position += self.qe[-1]
            self.current_inventory_position += self.qr[-1]
            
        elif self.dual_index:
            self.current_inventory_position_e -= self.current_demand
            self.current_inventory_position_e += self.qe[-1]
            self.current_inventory_position_e += \
            self.qr[-self.lead_time_difference-1]
        
            self.current_inventory_position_r -= self.current_demand
            self.current_inventory_position_r += self.qe[-1]
            self.current_inventory_position_r += self.qr[-1]
                        
        self.cost_update()

        self.current_inventory += self.current_qe + self.current_qr - \
                                  self.current_demand
                                                           
        self.inventory.append(self.current_inventory)
    
        if self.single_index:
            self.inventory_position.append(self.current_inventory_position)
            
        elif self.dual_index:
            self.inventory_position_e.append(self.current_inventory_position_e)
            self.inventory_position_r.append(self.current_inventory_position_r)

    def cost_update(self):
        self.current_cost = self.cost_e * self.qe[-1] + \
                            self.cost_r * self.qr[-1] + \
                            self.holding_cost * \
                            max(0, self.current_inventory + \
                            self.current_qe + \
                            self.current_qr - \
                            self.current_demand) + \
                            self.shortage_cost * \
                            max(0, self.current_demand - \
                            self.current_inventory - \
                            self.current_qe - \
                            self.current_qr)
        
        self.cost.append(self.current_cost)
                    
    def calculate_total_cost(self):
        
        self.total_cost = sum(self.cost)
        
    def capped_dual_index_sum(self, k):
        
        """ 
        Implementation of Eq. (3) Sun, J., & Van Mieghem, J. A. (2019). 
        Robust dual sourcing inventory management: Optimality of capped dual 
        index policies and smoothing. Manufacturing & Service 
        Operations Management, 21(4), 912-931.
        """
        
        Itk = self.current_inventory
        Itk += sum(self.qr[-self.lead_time_r+i] for i in range(k+1))
        if self.lead_time_e > max(1,self.lead_time_e-k):
            Itk += sum(self.qe[-self.lead_time_e+i] for i in \
                   range(min(k,self.lead_time_e-1)+1))

        return Itk
        
    def simulate(self):
        """ 
        Simulate dual sourcing dynamics. Each period consists of the following
        stages: (1) orders are placed, (2) shipments are received,
        (3) demand is revealed, (4) inventory and costs are updated.
        """
        
        for t in range(self.period):

            # (1) place orders
            if self.single_index:
                if self.current_inventory_position < self.target_order_level_e:
                    self.qe.append(max(0,self.current_demand-self.Delta))
                else:
                    self.qe.append(0)
            
                self.qr.append(min(self.Delta,self.current_demand))
            
            elif self.dual_index:
                self.qe.append(max(0,self.target_order_level_e-\
                               self.current_inventory_position_e-\
                               self.qr[-self.lead_time_difference]))
                self.qr.append(self.target_order_level_r-\
                               self.current_inventory_position_r-self.qe[-1])
            
            elif self.tailored_base_surge:
                self.qe.append(max(0,self.single_base_stock_level \
                                   -self.current_inventory \
                                   -self.regular_order_Q))
                self.qr.append(self.regular_order_Q)
                
            elif self.capped_dual_index:
                Itt = self.capped_dual_index_sum(0)
                ItLm1 = self.capped_dual_index_sum(self.lead_time_difference-1)
                
                if self.demand_flag == 1:
                    self.qe.append(int(max(0,self.s_e_optimal[t]-Itt)))
                    self.qr.append(int(min(max(0,self.s_r_optimal[t]-ItLm1),self.q_r_ast[t])))
                else:
                    self.qe.append(max(0,self.s_e_optimal-Itt))
                    self.qr.append(min(max(0,self.s_r_optimal-ItLm1),self.q_r_ast))

            # (2) receive shipments
            self.current_qe = self.qe[-self.lead_time_e-1]
            self.current_qr = self.qr[-self.lead_time_r-1]
            
            # (3) reveal demand
            if self.demand_flag == -1:
            	self.current_demand = self.demand_generator()
            else:
            	self.current_demand = self.demand_generator(t)
            	
            self.demand.append(self.current_demand)
        
            # (4) update inventory and costs
            self.inventory_evolution()
        
        # calculate difference between regular target order level
        # and inventory level
        if self.single_index:
            self.single_index_D_Delta = self.target_order_level_r - \
                                        self.current_inventory
        elif self.dual_index:
            self.dual_index_G_Delta = self.target_order_level_e - \
                                      self.current_inventory
                                                
        self.calculate_total_cost()

def single_index_zr_Delta(samples,
                          Delta_arr,
                          ce=0, 
                          cr=0, 
                          le=0, 
                          lr=0,
                          h=0, 
                          b=0,
                          T=200,
                          zr=0):
    """ 
    This function calculates the single index regular target order level zr
    and corresponding target order level difference Delta
    (for more information, see Scheller-Wolf, A., Veeraraghavan, S., 
    & van Houtum, G. J. (2007). Effective dual sourcing with a single 
    index policy. Working Paper, Tepper School of Business, 
    Carnegie Mellon University, Pittsburgh.)
    
    Parameters: 
    samples (int): number of samples
    Delta_arr (list): list of target order level differences
    ce (int): per unit cost of expedited supply
    cr (int): per unit cost of regular supply 
    le (int): expedited supply lead time
    lr (int): regular supply lead time
    h (int): holding cost per unit
    b (int): shortage cost per unit 
    T (int): number of periods
    zr (int): regular target order level
    
    Returns:
    optimal_zr (int), optimal_Delta (int): optimal regular single index 
    target order level and target order level difference
  
    """
    zr_arr = []
    for Delta in Delta_arr:
        
        D_Delta_arr = []
        
        for i in range(samples):
            S = DualSourcingModel(ce=ce, 
                                  cr=cr, 
                                  le=le, 
                                  lr=lr, 
                                  h=h, 
                                  b=b,
                                  T=T,
                                  I0=zr,
                                  zr=zr,
                                  Delta=Delta,
                                  single_index=True)
        
            S.simulate()  
            D_Delta_arr.append(S.single_index_D_Delta)
    
        sort = sorted(D_Delta_arr)
        zr_ = sort[int(len(D_Delta_arr) * S.critical_fractile)]
        zr_arr.append(zr_)
            
    cost_arr_mean = []
    cost_arr_std = []
    for i in range(len(Delta_arr)):
        cost_tmp = []
    
        for j in range(samples):
                S = DualSourcingModel(ce=ce, 
                                      cr=cr, 
                                      le=le, 
                                      lr=lr, 
                                      h=h, 
                                      b=b,
                                      T=T, 
                                      I0=zr_arr[i],
                                      zr=zr_arr[i],
                                      Delta=Delta_arr[i],
                                      single_index=True)
            
                S.simulate()
                cost_tmp.append(S.total_cost/T)
    
        cost_arr_mean.append(np.mean(cost_tmp))
        cost_arr_std.append(np.std(cost_tmp, ddof=1))
        
    Delta_arr = np.asarray(Delta_arr)
    zr_arr = np.asarray(zr_arr)
    cost_arr_mean = np.asarray(cost_arr_mean)
    
    optimal_Delta = Delta_arr[cost_arr_mean == min(cost_arr_mean)][0]   
    optimal_zr = zr_arr[cost_arr_mean == min(cost_arr_mean)][0]
    
    print("costs (mean/std):", cost_arr_mean, cost_arr_std)
    print("Delta*:", optimal_Delta)
    print("z_r*:", optimal_zr)
    
    return optimal_zr, optimal_Delta

def dual_index_ze_Delta(samples,
                        Delta_arr,
                        ce=0, 
                        cr=0, 
                        le=0, 
                        lr=0,
                        h=0, 
                        b=0,
                        T=200,
                        ze=0):
    """ 
    This function calculates the dual index expedited target order level ze
    and corresponding target order level difference Delta
    (for more information, see Veeraraghavan, S., & Scheller-Wolf, A. 
    (2008). Now or later: A simple policy for effective dual sourcing 
    in capacitated systems. Operations Research, 56(4), 850-864.)
    
    Parameters: 
    samples (int): number of samples
    Delta_arr (list): list of target order level differences
    ce (int): per unit cost of expedited supply
    cr (int): per unit cost of regular supply 
    le (int): expedited supply lead time
    lr (int): regular supply lead time
    h (int): holding cost per unit
    b (int): shortage cost per unit 
    T (int): number of periods
    ze (int): expedited target order level
    
    Returns:
    optimal_ze (int), optimal_Delta (int): optimal expedited dual index 
    target order level and target order level difference
  
    """
    ze_arr = []
    for Delta in Delta_arr:
        
        G_Delta_arr = []
        
        for i in range(samples):
            S = DualSourcingModel(ce=ce, 
                                  cr=cr, 
                                  le=le, 
                                  lr=lr, 
                                  h=h, 
                                  b=b,
                                  T=T,
                                  I0=ze,
                                  ze=ze,
                                  Delta=Delta,
                                  dual_index=True)
        
            S.simulate()  
            G_Delta_arr.append(S.dual_index_G_Delta)
    
        sort = sorted(G_Delta_arr)
        ze_ = sort[int(len(G_Delta_arr) * S.critical_fractile)]
        ze_arr.append(ze_)
            
    cost_arr_mean = []
    cost_arr_std = []
    for i in range(len(Delta_arr)):
        cost_tmp = []
    
        for j in range(samples):
                S = DualSourcingModel(ce=ce, 
                                      cr=cr, 
                                      le=le, 
                                      lr=lr, 
                                      h=h, 
                                      b=b,
                                      T=T, 
                                      I0=ze_arr[i],
                                      ze=ze_arr[i],
                                      Delta=Delta_arr[i],
                                      dual_index=True)
            
                S.simulate()
                cost_tmp.append(S.total_cost/T)
    
        cost_arr_mean.append(np.mean(cost_tmp))
        cost_arr_std.append(np.std(cost_tmp, ddof=1))
        
    Delta_arr = np.asarray(Delta_arr)
    ze_arr = np.asarray(ze_arr)
    cost_arr_mean = np.asarray(cost_arr_mean)
    
    optimal_Delta = Delta_arr[cost_arr_mean == min(cost_arr_mean)][0]   
    optimal_ze = ze_arr[cost_arr_mean == min(cost_arr_mean)][0]
    
    print("costs (mean/std):", cost_arr_mean, cost_arr_std)
    print("Delta*:", optimal_Delta)
    print("z_e*:", optimal_ze)
    
    return optimal_ze, optimal_Delta

def tailored_base_surge_Q_S(Q_arr,
                            s_arr,
                            ce=0, 
                            cr=0, 
                            le=0, 
                            lr=0,
                            h=0, 
                            b=0,
                            T=200):
    """ 
    This function calculates the dual index expedited target order level ze
    and corresponding target order level difference Delta
    (for more information, see Allon, G., & Van Mieghem, J. A. (2010). 
    Global dual sourcing: Tailored base-surge allocation to near- 
    and offshore production. Management Science, 56(1), 110-124. and
    Chen, B., & Shi, C. (2019). Tailored base-surge policies in 
    dual-sourcing inventory systems with demand learning. 
    Available at SSRN 3456834.)
    
    Parameters: 
    Q_arr (list): list of regular order quantities
    s_arr (list): list of single base-stock target level
    ce (int): per unit cost of expedited supply
    cr (int): per unit cost of regular supply 
    le (int): expedited supply lead time
    lr (int): regular supply lead time
    h (int): holding cost per unit
    b (int): shortage cost per unit 
    T (int): number of periods
    
    Returns:
    optimal_Q (int), optimal_S (int): optimal tailored base-surge parameters
  
    """
            
    min_cost = 1e9
    
    optimal_Q = 0
    optimal_s = 0
    for Q in Q_arr:
        for s in s_arr:
                    
            S = DualSourcingModel(ce=ce, 
                                  cr=cr, 
                                  le=le, 
                                  lr=lr, 
                                  h=h, 
                                  b=b,
                                  T=T, 
                                  I0=s,
                                  Q=Q,
                                  s=s,
                                  tailored_base_surge=True)
        
            S.simulate()
            cost_tmp = S.total_cost/T
        
            if cost_tmp < min_cost:
                optimal_Q = Q
                optimal_s = s
                min_cost = cost_tmp
                        
    print("minimum cost:", min_cost)
    print("Q*:", optimal_Q)
    print("s*:", optimal_s)
    
    return optimal_Q, optimal_s

def capped_dual_index_parameters(u1_arr,
                                 u2_arr,
                                 u3_arr,
                                 ce=0, 
                                 cr=0, 
                                 le=0, 
                                 lr=0,
                                 h=0, 
                                 b=0,
                                 T=200,
                                 demand_distribution=[-1]):
    """ 
    This function calculates the dual index expedited target order level ze
    and corresponding target order level difference Delta
    (for more information, see Sun, J., & Van Mieghem, J. A. (2019). 
    Robust dual sourcing inventory management: Optimality of capped dual 
    index policies and smoothing. Manufacturing & Service 
    Operations Management, 21(4), 912-931.)
    
    Parameters: 
    u1_arr (list): list of capped dual index parameter 1
    u2_arr (list): list of capped dual index parameter 2
    u3_arr (list): list of capped dual index parameter 3
    ce (int): per unit cost of expedited supply
    cr (int): per unit cost of regular supply 
    le (int): expedited supply lead time
    lr (int): regular supply lead time
    h (int): holding cost per unit
    b (int): shortage cost per unit 
    T (int): number of periods
    demand_distribution (array): demand distribution that is given
    by an array ([-1] indicates that the standard, uniform 
    distribution should be used)
    
    Returns:
    optimal_Q (int), optimal_S (int): optimal tailored base-surge parameters
  
    """
            
    min_cost = 1e9
    
    optimal_u1 = 0
    optimal_u2 = 0
    optimal_u3 = 0
    for u1 in u1_arr:
        for u2 in u2_arr:
            for u3 in u3_arr:
                if demand_distribution[0] == -1:
                    S = DualSourcingModel(ce=ce, 
		                                  cr=cr, 
    		                              le=le, 
    		                              lr=lr, 
    		                              h=h, 
    		                              b=b,
    		                              T=T, 
    		                              I0=0,
    		                              u1=u1,
    		                              u2=u2,
    		                              u3=u3,
    		                              capped_dual_index=True)
                else:
                    S = DualSourcingModel(ce=ce, 
    		                              cr=cr, 
    		                              le=le, 
    		                              lr=lr, 
    		                              h=h, 
    		                              b=b,
    		                              T=T, 
    		                              I0=0,
    		                              u1=u1,
    		                              u2=u2,
    		                              u3=u3,
    		                              capped_dual_index=True,
    		                              demand_distribution=demand_distribution)
            
                S.simulate()
                cost_tmp = S.total_cost/T
            
                if cost_tmp < min_cost:
                    optimal_u1 = u1
                    optimal_u2 = u2
                    optimal_u3 = u3
                    min_cost = cost_tmp
                    
    print("minimum cost:", min_cost)
    print("u1*:", optimal_u1)
    print("u2*:", optimal_u2)
    print("u3*:", optimal_u3)

    return optimal_u1, optimal_u2, optimal_u3
