import numpy as np
from itertools import product

class SingleSourcingModel:
    def __init__(self, 
                 l=0, 
                 h=0, 
                 b=0, 
                 r=0,
                 T=200,
                 I0=0,
                 optimal_base_stock=True):
        """ 
        Initialization of dual sourcing model. 
        
        Parameters: 
        l (int): lead time
        h (int): holding cost per unit
        b (int): shortage cost per unit
        r (int): target order level
        T (int): number of periods
        I0 (int): initial inventory level

        """

        self.lead_time = l
        self.holding_cost = h
        self.shortage_cost = b     
        
        self.current_demand = 0
        self.current_cost = 0

        # current order quantities
        self.current_q = 0
        
        if self.lead_time == 0:
            self.q = [self.current_q]
        else:
            self.q = self.lead_time*[self.current_q]
            
        # simulation period and containers
        self.period = T

        self.cost = [self.current_cost]
        self.demand = [self.current_demand]
        self.total_cost = 0
        
        self.demand_support = [0,1,2,3,4]
        self.demand_distribution = lambda: np.random.choice(self.demand_support)
        
        self.critical_fractile = self.shortage_cost/(self.holding_cost+\
                                                    self.shortage_cost)
        
        if optimal_base_stock:
            sort = sorted([sum([self.demand_distribution() for j in \
                                range(self.lead_time+1)]) for i in range(100000)])

            self.target_inventory_position_level = sort[int(len(sort) * \
                                                   self.critical_fractile)]
            print(self.target_inventory_position_level)

        else:
            self.target_inventory_position_level = r + 1   
            
        self.current_inventory = self.target_inventory_position_level
        self.current_inventory_position = self.current_inventory
        self.inventory = [self.current_inventory]
        self.inventory_position = [self.current_inventory_position] 
        
    def inventory_evolution(self):
        """ 
        Update inventory position, inventory, and cost.
        """
        self.current_inventory_position += self.q[-1] - self.current_demand
            
        self.cost_update()

        self.current_inventory += self.current_q - self.current_demand
                                                           
        self.inventory.append(self.current_inventory)
        self.inventory_position.append(self.current_inventory_position)
            
    def cost_update(self):
        self.current_cost = self.holding_cost * \
                            max(0, self.current_inventory + \
                            self.current_q - \
                            self.current_demand) + \
                            self.shortage_cost * \
                            max(0, self.current_demand - \
                            self.current_inventory - \
                            self.current_q)  
                                
        self.cost.append(self.current_cost)
                    
    def calculate_total_cost(self):
        
        self.total_cost = sum(self.cost)
        
    def calculate_optimal_cost(self):
        
        expected_backorder = 0
        prob = (1/len(self.demand_support))**(self.lead_time+1)
        demands = product(self.demand_support, repeat=self.lead_time+1)
        for demand in demands:
            sum_demand = sum(demand)
            if sum_demand >= self.target_inventory_position_level:
                expected_backorder += sum_demand - \
                    self.target_inventory_position_level 
        
        expected_backorder *= prob        
        
        self.optimal_cost = self.holding_cost * \
            ( self.target_inventory_position_level - \
             (self.lead_time+1) * np.mean(self.demand_support) ) + \
            (self.holding_cost + self.shortage_cost)*expected_backorder
        
    def simulate(self):
        """ 
        Simulate single sourcing dynamics. Each period consists of the following
        stages: (1) order is placed, (2) shipments are received,
        (3) demand is revealed, (4) inventory and costs are updated.
        """
        
        for t in range(self.period):
            
            # (1) place order
            if self.current_inventory_position < \
                self.target_inventory_position_level:
                self.q.append(self.target_inventory_position_level - \
                              self.current_inventory_position)
            else:
                self.q.append(0)
            
            # (2) receive shipments
            self.current_q = self.q[-self.lead_time-1]
            
            # (3) reveal demand
            self.current_demand = self.demand_distribution()
            self.demand.append(self.current_demand)
        
            # (4) update inventory and costs
            self.inventory_evolution()
                                                
        self.calculate_total_cost()
        
        self.calculate_optimal_cost()
