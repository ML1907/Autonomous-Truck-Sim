# Different traffic situations
import numpy as np
from casadi import *

from helpers import *

class trailing:
    '''
    The ego vehicle keeps lane and adapts to leading vehicle speed. Defines longitudinal safety constraint S() that will be then
    set as mpc constraint within the makeController class.
    '''
    def __init__(self,vehicle,N,min_distx = 5, lanes = 3, laneWidth = 6.5,v_legal = 60/3.6):
        self.name = 'trailing'
        self.N = N
        self.vehicle = vehicle
        self.nx,self.nu,_,_ = vehicle.getSystemDim()
        self.egoWidth, self.egoLength,self.L_tract, self.L_trail = vehicle.getSize()

        # Road definitions
        self.lanes = lanes
        self.laneWidth = laneWidth

        self.vmax = v_legal+5/3.6

        self.Time_headway = 0.5

        self.min_distx = min_distx   # Minimum longitudinal safety distance
        self.p = MX.sym('p',1,N+1)   # Symbolic x position variable to define safety distance over MPC horizon

        self.egoPx = []              # real time x position of ego vehicle
        self.egoPy = []

    def getReference(self,refx,refu):
        # Returns state and input reference for all steps in horizon 
        refx_out = DM(self.nx,self.N+1)
        refu_out = DM(self.nu,self.N)
        
        for i in range(self.nx):
            refx_out[i,:] = refx[i]
        for i in range(self.nu):
            refu_out[i,:] = refu[i]

        return refx_out, refu_out

    def constraint(self,traffic,opts):
        '''Computes the safety distance constraint based on lead vehicle.'''
        leadWidth, leadLength = traffic.getVehicles()[0].getSize()
        idx = self.getLeadVehicle(traffic)
        if len(idx) == 0:
            dist_t = 0
        else:
            v0_idx = traffic.getVehicles()[idx[0]].v0
            dist_t = v0_idx * self.Time_headway

        safeDist = self.min_distx + leadLength/2 + self.L_tract
        return Function('S',[self.p],[self.p-safeDist],['p'],['D_min'])

    def getRoad(self):
        roadMax = 2*self.laneWidth
        roadMin = -(self.lanes-2)*self.laneWidth
        laneCenters = [self.laneWidth/2,self.laneWidth*3/2,-self.laneWidth*1/2]

        return roadMin, roadMax, laneCenters

    def setEgoLane(self):
        '''Determines the ego vehicle's lane based on its current Y-position.'''
        x = self.vehicle.getPosition()
        self.egoPx = x[0]
        self.egoPy = x[1]
        if self.egoPy < 0:
            self.egoLane = -1
        elif self.egoPy > self.laneWidth:
            self.egoLane = 1
        else:
            self.egoLane = 0

    def getEgoLane(self):
        return self.egoLane
    
    def getLeadVehicle(self,traffic):
        '''Finds the closest vehicle ahead in the same lane.'''
        self.setEgoLane()
        reldistance = 10000             # A large number
        leadInLane = []
        i = 0
        for vehicle in traffic.vehicles:
            if self.egoLane == vehicle.getLane():
                if vehicle.getState()[0] > self.egoPx:
                    distance = vehicle.getState()[0] - self.egoPx
                    if distance < reldistance:
                        leadInLane = [i]
                        reldistance = distance
            i += 1
        return leadInLane

class simpleOvertake:
    '''
    The ego vehicle overtakes the lead vehicle
    '''
    def __init__(self,vehicle,N, min_distx = 5, lanes = 3, laneWidth = 6.5,v_legal = 60/3.6):
        self.name = 'simpleOvertake'
        self.N = N
        self.nx,self.nu,_,_ = vehicle.getSystemDim()

        # Road definitions
        self.lanes = lanes
        self.laneWidth = laneWidth

        self.vmax = v_legal+5/3.6

        self.Time_headway = 0.5

        # Ego vehicle dimensions
        self.egoWidth, self.egoLength,self.L_tract, self.L_trail = vehicle.getSize()
        
        # Safety constraint definitions
        self.min_distx = min_distx
        self.pxL = MX.sym('pxL',1,N+1)    # ego vehicle's predicted X-position specifically during a lane-change maneuver
        self.px = MX.sym('x',1,N+1)       # ego vehicle's predicted X-position along the road throughout the prediction horizon

        self.traffic_x = MX.sym('x',1,N+1)  #Predicted x position of surrounding vehicles over the entire prediction horizon N
        self.traffic_y = MX.sym('y',1,N+1)
        self.traffic_sign = MX.sym('sign',1,N+1)   # Encodes the direction (+1 for right, -1 for left) of the lane shift
        self.traffic_shift = MX.sym('shift',1,N+1)  # Represents the lateral displacement (y-offset) to reach the target lane


    def getReference(self,refx,refu):
        # Returns state and input reference for all steps in horizon 
        refx_out = DM(self.nx,self.N+1)
        refu_out = DM(self.nu,self.N)
        
        for i in range(self.nx):
            refx_out[i,:] = refx[i]
        for i in range(self.nu):
            refu_out[i,:] = refu[i]

        return refx_out,refu_out

    def getRoad(self):
        roadMax = 2*self.laneWidth
        roadMin = -(self.lanes-2)*self.laneWidth
        laneCenters = [self.laneWidth/2,self.laneWidth*3/2,-self.laneWidth*1/2]
        
        return roadMin, roadMax, laneCenters

    def constraint(self,traffic,opts):
        constraints = []
        leadWidth, leadLength = traffic.getVehicles()[0].getSize()
        for i in range(traffic.getDim()):
            v0_i = traffic.vehicles[i].v0
            func1 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
                    tanh(self.px - self.traffic_x + leadLength/2 + self.L_tract + v0_i * self.Time_headway + self.min_distx )  + self.traffic_shift/2
            func2 = self.traffic_sign * (self.traffic_sign*(self.traffic_y-self.traffic_shift) + self.egoWidth + leadWidth) / 2 * \
                    tanh( - (self.px - self.traffic_x)  + leadLength/2 + self.L_trail + v0_i * self.Time_headway+ self.min_distx )  + self.traffic_shift/2

            constraints.append(Function('S',[self.px,self.traffic_x,self.traffic_y,
                                    self.traffic_sign,self.traffic_shift,],
                                    [func1+func2],['px','t_x','t_y','t_sign','t_shift'],['y_cons']))
        return constraints