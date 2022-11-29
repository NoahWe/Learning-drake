import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display, SVG
from scipy.linalg import expm
import pydot
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, LinearQuadraticRegulator,
                        Parser, Saturation, Simulator, WrapToSystem, PlanarSceneGraphVisualizer, 
                        Linearize, DiscreteTimeLinearQuadraticRegulator)    
from matplotlib import pyplot as plt

from pydrake.systems.primitives import LinearSystem, MatrixGain, LogVectorOutput, Adder, ConstantVectorSource

from pydrake.autodiffutils import AutoDiffXd, InitializeAutoDiff, ExtractGradient
from pydrake.systems.framework import BasicVector_
from tqdm import tqdm as tqdm


class getGradients:
    def __init__(self, eq_state, eq_input, urdf):
        
        # Point to linearize around
        self.eq_state = eq_state
        self.eq_input = eq_input

        # Initialize parameters for making the multibodyplant
        self.builder = DiagramBuilder()
        self.urdf = urdf
        self.plant = None
        self.scene_graph = None
        self.context = None
        self.time_step = None

        # Initialize parameters for linearization
        self.plant_ad = None
        self.context_ad = None
        self.A = None
        self.B = None
        self.gravity = None
        self.coriolis = None
        self.lqr_gain = None
    
    def makePlantFromURDF(self):

        # Initiate diagram and make multibody plant using Acrobot.urdf
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.0)  # Add a scene graph to the diagram
        Parser(self.plant).AddModelFromFile(self.urdf)  # Parse robot URDF file
        self.plant.Finalize()  # Finalize the acrobot plant

        # Set context
        self.context = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(self.context, self.eq_state)
        self.plant.get_actuation_input_port().FixValue(self.context, self.eq_input)
    
    def getGradients(self):
        # Start of Linearization with AutoDiff
        # AutoDiffXd quantities have a value and derivative part

        # Set up autodiff system
        self.plant_ad = self.plant.ToScalarType[AutoDiffXd]()  # Convert regular plant to AutoDiff plant
        self.context_ad = self.plant_ad.CreateDefaultContext()   # Convert regular context to AutoDiff context
        xu_val = np.hstack((self.eq_state, self.eq_input))             # make a vector with all initial states and inputs
        states = InitializeAutoDiff(xu_val)              # Initialize autodiff with the state+input vector

        nstates = self.context_ad.num_continuous_states()     # Get number of states to separate inputs and states later
        x_ad = states[:nstates]                          # Separate states from the state+input vector
        u_ad = states[nstates:]                          # Separate inputs from the state+input vector
            
        self.plant_ad.SetPositionsAndVelocities(self.context_ad, x_ad)

        self.plant_ad.get_actuation_input_port().FixValue(self.context_ad, BasicVector_[AutoDiffXd](u_ad))

        # Derivatives
        xdot_ad = self.plant_ad.EvalTimeDerivatives(self.context_ad).CopyToVector()  # Evaluate time derivatives and copy to vector        
        AB = ExtractGradient(xdot_ad)  # Get the gradient from the time derivatives

        self.A = AB[:, :nstates]  # Get the linearized A matrix from the time derivative gradients
        self.B = AB[:, nstates:]  # Get the linearized B matrix from the time derivative gradients


    def getA(self):
        return self.linear.A(), self.linear.B()


    def lqr_controller(self, Q, R):
        self.lqr_gain = -LinearQuadraticRegulator(self.A, self.B, Q, R)[0]
        self.lqr_gain = MatrixGain(self.lqr_gain)
        
class makeAcrobot(getGradients):
    
    def __init__(self):
        pass

    def wire(self):
        self.lqr = self.builder.AddSystem(self.lqr_gain)
        self.input_actuation_index = self.plant.get_actuation_input_port().get_index()
        self.output_state_index = self.plant.get_state_output_port().get_index()
        # Add Input saturation and angle wrapping to the system
        self.saturation = self.builder.AddSystem(Saturation(min_value=[-10], max_value=[10]))
        self.builder.Connect(self.saturation.get_output_port(), self.acrobot.get_actuation_input_port())
        wrapangles = WrapToSystem(4)
        wrapangles.set_interval(0, 0, 2. * np.pi)
        wrapangles.set_interval(1, -np.pi, np.pi)
        self.wrapto = self.builder.AddSystem(wrapangles)
        
        self.lqr_gain(Q, R)

        self.adder = self.builder.AddSystem(Adder(2, 4))
        self.vector = self.builder.AddSystem(ConstantVectorSource(-eq_state))

        # Wire connections
        self.builder.Connect(self.acrobot.get_state_output_port(), self.wrapto.get_input_port())
        self.builder.Connect(self.wrapto.get_output_port(), self.adder.get_input_port(0))
        self.builder.Connect(self.vector.get_output_port(), self.adder.get_input_port(1))
        self.builder.Connect(self.adder.get_output_port(), self.lqr.get_input_port())
        self.builder.Connect(self.lqr.get_output_port(), self.saturation.get_input_port())
        
        self.logger_u = LogVectorOutput(self.lqr.get_output_port(), self.builder)
        self.logger_i = LogVectorOutput(self.wrapto.get_output_port(), self.builder)
        self.logger_state = LogVectorOutput(self.acrobot.get_state_output_port(), self.builder)
        self.logger_saturation = LogVectorOutput(self.saturation.get_output_port(), self.builder)

        # finish building the block diagram
        self.diagram = self.builder.Build()
        self.diagram.set_name("diagram")

        display(SVG(pydot.graph_from_dot_data(
            self.diagram.GetGraphvizString(max_depth=2))[0].create_svg()))
    
    def visuals(self):
        # add a visualizer and wire it
        self.visualizer = self.builder.AddSystem(PlanarSceneGraphVisualizer(self.scene_graph, xlim=[-3., 3.], ylim=[-3, 3.], show=True))
        self.builder.Connect(self.scene_graph.get_query_output_port(), self.visualizer.get_input_port(0))

    def simulate(self, x0, sim_time=5.):
        self.simulator = Simulator(self.diagram)

        if "self.visualizer" in locals():
            self.visualizer.start_recording()
    
        # reset initial time and state
        self.context = self.simulator.get_mutable_context()
        self.context.SetTime(0.)
        self.context.SetContinuousState(x0)

        # run sim
        self.simulator.Initialize()
        self.simulator.AdvanceTo(sim_time)

        if "self.visualizer" in locals():
            # stop video
            self.visualizer.stop_recording()    

            # construct animation
            ani = self.visualizer.get_recording_as_animation()

            # display animation below the cell
            display(HTML(ani.to_jshtml()))

            # reset to empty video
            self.visualizer.reset_recording()
        
    def retrieve_log(self):
        sim_context = self.simulator.get_context()
        return self.logger_i.FindLog(sim_context), self.logger_u.FindLog(sim_context), \
                self.logger_state.FindLog(sim_context), self.logger_saturation.FindLog(sim_context)

if __name__ == "__main__": 
    
    Q = np.diag((10., 10., 1., 1.))
    R = np.array([1.])

    eq_state = np.array([np.pi, 0., 0., 0.])
    eq_input = np.array([0.])

    C = np.eye(4)
    D = np.zeros((4, 1))

    # Initialize acrobots
    acrobot_autodiff = getGradients(eq_state, eq_input, "Acrobot/acrobot_tmotors.urdf")
    acrobot_autodiff.makePlantFromURDF()
    acrobot_autodiff.getGradients()
    # print(acrobot_autodiff.gravity, acrobot_autodiff.coriolis)
    acrobot_autodiff.lqr_controller(Q, R)
    # print(acrobot_autodiff.lqr_gain.D())
    print(acrobot_autodiff.A)

    # print(f[eigenvalues[eigenvalues.all()]])
    

    # acrobot_urdf = Acrobot(eq_state, eq_input, "Acrobot/Acrobot.urdf")

    # # Connect diagrams
    # acrobot_autodiff.wire(Q, R, "autodiff")
    # acrobot_urdf.wire(Q, R, "system")

    # x0 = np.array([[0.99*np.pi], [0.1], [0.], [0.]])

    # acrobot_autodiff.simulate(x0)
    # acrobot_urdf.simulate(x0)

    # i_autodiff, u_autodiff, state_autodiff, sat_autodiff = acrobot_autodiff.retrieve_log()
    # i_urdf, u_urdf, state_urdf, sat_urdf = acrobot_urdf.retrieve_log()

    # t_autodiff = np.linspace(0, 5, np.shape(state_autodiff.data())[1])
    # t_urdf = np.linspace(0, 5, np.shape(state_urdf.data())[1])
    
    # plt.close()
    # plt.plot(t_autodiff, i_autodiff.data()[0, :], label="autodiff")
    # plt.plot(t_urdf, i_urdf.data()[0, :], label="urdf")
    # plt.title("Input to LQR gains")
    # plt.legend()
    # plt.show()