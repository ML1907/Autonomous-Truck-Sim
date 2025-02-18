# Packages
from casadi import *
import numpy as np
import pandas as pd

# Classes and helpers
from vehicleModelGarage import vehBicycleKinematic
from scenarios import trailing, simpleOvertake
from traffic import vehicleSUMO, combinedTraffic
from controllers import makeController, makeDecisionMaster
from helpers import *

from templateRLagent import RLAgent

# Set Gif-generation
makeMovie = False
directory = r"C:\Users\A521105\Desktop\Volvo\Visualization\simRes.gif"

# System initialization 
dt = 0.2                    # Simulation time step (Impacts traffic model accuracy)
f_controller = 1            # Controller update frequency, i.e updates at each t = dt*f_controller
N =  30                     # MPC Horizon length

ref_vx = 70/3.6             # Higway speed limit in (m/s)

# -------------------------- Initilize RL agent object ----------------------------------
# The agent is feed to the decision maker, changing names requries changing troughout code base
RL_Agent = RLAgent()

# ----------------- Ego Vehicle Dynamics and Controller Settings ------------------------
vehicleADV = vehBicycleKinematic(dt,N)

vehWidth,vehLength,L_tract,L_trail = vehicleADV.getSize()
nx,nu,nrefx,nrefu = vehicleADV.getSystemDim()

# Integrator
int_opt = 'rk'
vehicleADV.integrator(int_opt,dt)
F_x_ADV  = vehicleADV.getIntegrator()

# Set Cost parameters
Q_ADV = [0,40,3e2,5,5]                           # State cost, Entries in diagonal matrix
R_ADV = [5,5]                                    # Input cost, Entries in diagonal matrix
q_ADV_decision = 50

vehicleADV.cost(Q_ADV,R_ADV)
vehicleADV.costf(Q_ADV)
L_ADV,Lf_ADV = vehicleADV.getCost()

# ------------------ Problem definition ---------------------
scenarioTrailADV = trailing(vehicleADV,N,lanes = 3,v_legal = ref_vx)
scenarioADV = simpleOvertake(vehicleADV,N,lanes = 3,v_legal = ref_vx)
roadMin, roadMax, laneCenters = scenarioADV.getRoad()
    
# -------------------- Traffic Set up -----------------------
# * Be carful not to initilize an unfeasible scenario where a collsion can not be avoided
# # Initilize ego vehicle
vx_init_ego = 50/3.6                                # Initial velocity of the ego vehicle
disable_ego_interaction = True                      # Set to True to IGNORE ego vehicle (by agents)
vehicleADV.setInit([0,laneCenters[0]],vx_init_ego)

# # Initilize surrounding traffic
# Lanes [0,1,2] = [Middle,left,right]
advVeh1 = vehicleSUMO(dt,N,[30,laneCenters[1]],[0.75*ref_vx,0],type = "normal",disable_ego_interaction=disable_ego_interaction)
advVeh2 = vehicleSUMO(dt,N,[40,laneCenters[0]],[0.7*ref_vx,0],type = "normal",disable_ego_interaction=disable_ego_interaction)
advVeh3 = vehicleSUMO(dt,N,[100,laneCenters[2]],[0.65*ref_vx,0],type = "passive",disable_ego_interaction=disable_ego_interaction)
advVeh4 = vehicleSUMO(dt,N,[-20,laneCenters[1]],[1*ref_vx,0],type = "aggressive",disable_ego_interaction=disable_ego_interaction)
advVeh5 = vehicleSUMO(dt,N,[60,laneCenters[2]],[1*ref_vx,0],type = "aggressive",disable_ego_interaction=disable_ego_interaction)

# # Combine choosen vehicles in list
vehList = [advVeh1,advVeh2,advVeh3,advVeh4,advVeh5]

# # Define traffic object
leadWidth, leadLength = advVeh1.getSize()
traffic = combinedTraffic(vehList,vehicleADV,N,f_controller)   
traffic.setScenario(scenarioADV)
Nveh = traffic.getDim()

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#      Formulate optimal control problem using opti framework
# -----------------------------------------------------------------
# -----------------------------------------------------------------
dt_MPC = dt*f_controller
# Version = [trailing,leftChange,rightChange]
opts1 = {"version" : "leftChange", "solver": "ipopt", "integrator":"rk"}
MPC1 = makeController(vehicleADV,traffic,scenarioADV,N,opts1,dt_MPC)
MPC1.setController()
# MPC1.testSolver(traffic)
changeLeft = MPC1.getFunction()

opts2 = {"version" : "rightChange", "solver": "ipopt", "integrator":"rk"}
MPC2 = makeController(vehicleADV,traffic,scenarioADV,N,opts2,dt_MPC)
MPC2.setController()
# MPC2.testSolver(traffic)
changeRight = MPC2.getFunction()

opts3 = {"version" : "trailing", "solver": "ipopt", "integrator":"rk"}
MPC3 = makeController(vehicleADV,traffic,scenarioTrailADV,N,opts3,dt_MPC)
MPC3.setController()
trailLead = MPC3.getFunction()

print("Initilization succesful.")

# Initilize Decision maker
decisionMaster = makeDecisionMaster(vehicleADV,traffic,[MPC1,MPC2,MPC3],
                                [scenarioTrailADV,scenarioADV],RL_Agent)

decisionMaster.setDecisionCost(q_ADV_decision)                  # Sets cost of changing decision

# # -----------------------------------------------------------------
# # -----------------------------------------------------------------
# #                         Simulate System
# # -----------------------------------------------------------------
# # -----------------------------------------------------------------
Respawn = False
tsim = 20                         # Total simulation time in seconds
Nsim = int(tsim/dt)              # Number of simulation steps (time steps) based on dt
tspan = np.linspace(0, tsim, Nsim)  # Time vector for plotting or analysis

# Initialize simulation
x_iter = DM(int(nx),1)
x_iter[:],u_iter = vehicleADV.getInit()
vehicleADV.update(x_iter,u_iter)

# Set reference trajectories for ego vehicle
refxADV = [0, laneCenters[1], ref_vx, 0, 0]   # Desired lane center & velocity
refxT_in, refxL_in, refxR_in = vehicleADV.setReferences(laneCenters, ref_vx)   # describe what the ego wants to do (simple target)

refu_in = [0,0,0]
refxT_out,refu_out = scenarioADV.getReference(refxT_in,refu_in)    # fully defined state and input references over time, generated by the scenario planner.
refxL_out,refu_out = scenarioADV.getReference(refxL_in,refu_in)
refxR_out,refu_out = scenarioADV.getReference(refxR_in,refu_in)

refxADV_out,refuADV_out = scenarioADV.getReference(refxADV,refu_in)

# Traffic states and prediction storage
x_lead = DM(Nveh, N+1)                        # Placeholder for lead vehicle positions (Nveh vehicles over horizon)
traffic_state = np.zeros((5, N+1, Nveh))     # Placeholder for traffic state predictions: [x, y, vx, vy, type]

# Data storage for plotting and analysis
X = np.zeros((nx, Nsim, 1))                  # Ego vehicle states over time
U = np.zeros((nu, Nsim, 1))                  # Ego vehicle inputs over time
X_pred = np.zeros((nx, N+1, Nsim))           # Predicted trajectories from MPC (no agents)

X_traffic = np.zeros((4, Nsim, Nveh))        # Traffic states (x, y, vx, theta) (agents)
X_traffic_ref = np.zeros((4, Nsim, Nveh))    # Traffic reference points (e.g., near and far points for lane following)

# Get initial traffic states
X_traffic[:, 0, :] = traffic.getStates()     # Record traffic states at time zero
testPred = traffic.prediction()              # traffic.prediction(): Predicts the states of all traffic vehicles (using IDM + MOBIL).
prediction_records = []                      # List to store prediction data for CSV
N_predictions = 10                           # Number of predictions to save (customizable)

# RL Feature Map
feature_map = np.zeros((5, Nsim, Nveh + 1))  # Store feature vectors for RL agent: ego + traffic

# # Simulation loop
for i in range(0,Nsim):
    # Get current traffic state
    x_lead[:,:] = traffic.prediction()[0,:,:].transpose()  # Extracts only the x positions of predicted vehicles to help the ego vehicle’s MPC
    traffic_state[:2,:,] = traffic.prediction()[:2,:,:]    # Store traffic predicted positions for the mpc constraints (N_pred=N*freqMPC samples)

    traffic_predictions = traffic.prediction()[:2, :N_predictions, :]  # Extract only x and y (N_predictions sample)
    # Store predictions for each vehicle at the current time step
    for j in range(Nveh):
        for pred_step in range(N_predictions):
            prediction_records.append([
                i,                  # time_step
                j,                  # vehicle_id
                pred_step,          # prediction_step
                traffic_predictions[0, pred_step, j],  # predicted_x
                traffic_predictions[1, pred_step, j]   # predicted_y
            ])

    # Initialize master controller
    if i % f_controller == 0:
        print("----------")
        print('Step: ', i)
        decisionMaster.storeInput([x_iter,refxL_out,refxR_out,refxT_out,refu_out,x_lead,traffic_state])

        # Update reference based on current lane
        refxL_out,refxR_out,refxT_out = decisionMaster.updateReference()

        # Compute optimal control action
        x_test,u_test,X_out = decisionMaster.chooseController()  #Executes the optimization. Waits to finish and then move on.
        u_iter = u_test[:,0]

    # Store results for plotting and analysis
    X[:, i] = x_iter              # Record ego vehicle state
    U[:, i] = u_iter              # Record ego vehicle input
    X_pred[:, :, i] = X_out       # Store the predicted trajectory from MPC

    # Simulate one step forward for the ego vehicle using the chosen input
    x_iter = F_x_ADV(x_iter, u_iter)
    vehicleADV.update(x_iter, u_iter)  # Update ego vehicle internal state

    # Store traffic states
    X_traffic[:, i, :] = traffic.getStates()         # Actual traffic states
    X_traffic_ref[:, i, :] = traffic.getReference()  # Traffic target lane positions

    # Simulate traffic forward
    traffic.update()                # Update traffic vehicles' states (IDM and MOBIL)
    if Respawn:
        traffic.tryRespawn(x_iter[0])   # Respawn traffic vehicles if needed (if behind ego)

print("Simulation finished")

i_crit = i

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#                    Plotting and data extraction
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# Creates animation of traffic scenario
if makeMovie:
    borvePictures(X,X_traffic,X_traffic_ref,vehList,X_pred,vehicleADV,scenarioADV,traffic,i_crit,f_controller,directory)


# -------------------------- Export X_traffic with Time Step and Vehicle ID --------------------------

# Each row corresponds to one vehicle at one time step. include time_step and vehicle_id columns
export_data = []
for i in range(Nsim):
    for j in range(Nveh):
        export_data.append([i, j, X_traffic[0, i, j], X_traffic[1, i, j], X_traffic[2, i, j], X_traffic[3, i, j]])

# Convert to DataFrame and Save to CSV
df_traffic = pd.DataFrame(export_data, columns=["time_step", "vehicle_id", "x", "y", "vx", "theta"])
df_traffic.to_csv('x_traffic.csv', index=False)

# -------------------------- Export X_traffic_predictions with same Time Step and Vehicle ID of X_traffic ------------

df_predictions = pd.DataFrame(prediction_records, columns=[
    "time_step", "vehicle_id", "prediction_step", "x", "y"
])
df_predictions.to_csv('x_traffic_predictions.csv', index=False)

# **Outcome:**
# - `x_traffic_predictions.csv` contains only `x` and `y` predictions.
# - Flexible with `N_predictions` for horizon length.
# - Long-format CSV suitable for easy analysis and plotting.


# -------------------------- Plot Traffic and Ego Trajectories --------------------------
def plot_trajectories(X_traffic, X):
    plt.figure(figsize=(10, 6))
    
    # Plot traffic vehicle trajectories
    for j in range(X_traffic.shape[2]):
        plt.plot(X_traffic[0, :, j], X_traffic[1, :, j], label=f'Traffic Vehicle {j+1}', linestyle='--')

    # Plot ego vehicle trajectory
    plt.plot(X[0, :, 0], X[1, :, 0], 'k--', label='Ego Vehicle', linewidth=2)

    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Traffic and Ego Vehicle Trajectories')
    plt.grid(True)
    plt.legend()
    plt.show()

# Call plotting function with original X_traffic format
plot_trajectories(X_traffic, X)