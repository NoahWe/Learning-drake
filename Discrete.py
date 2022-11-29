import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display, SVG
from scipy.linalg import expm
import pydot
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder,
                        Parser, Saturation, Simulator, WrapToSystem, PlanarSceneGraphVisualizer, 
                        Linearize, DiscreteTimeLinearQuadraticRegulator, LinearQuadraticRegulator)    
from matplotlib import pyplot as plt

from pydrake.systems.primitives import LinearSystem, MatrixGain, LogVectorOutput, Adder, ConstantVectorSource, ZeroOrderHold
from pydrake.systems.analysis import ResetIntegratorFromFlags

from tqdm import tqdm as tqdm


# copied from https://www.mwm.im/lqr-controllers-with-python/

import numpy as np
import scipy.linalg


def clqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u

    ref: Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    ref: Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals



def makePlant(eq_state, eq_input, urdf):
        """Makes plant of the system. Compatibility only tested for acrobot

        Args:
            eq_state (np.ndarray): System equilibrium point (states)
            eq_input (np.ndarray): System equilibrium point (input)
            urdf (string): Path to urdf file

        Returns:
            np.ndarray: A and B matrices of the linearized system
        """
        
        # Initiate diagram and make multibody plant using Acrobot.urdf
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)  # Add a scene graph to the diagram
        Parser(plant).AddModelFromFile(urdf)  # Parse robot URDF file
        plant.Finalize()  # Finalize the acrobot plant

        # Set context
        context = plant.CreateDefaultContext()
        context.get_mutable_continuous_state_vector().SetFromVector(eq_state)
        plant.get_actuation_input_port().FixValue(context, eq_input)

        # Linearize the system to get the continuous A and B matrices
        input_i = plant.get_actuation_input_port().get_index()
        output_i = plant.get_state_output_port().get_index()

        # Relaxed equilibrium check tolerance was necessary for DFKI acrobot (default=1e-6)
        # Must comment out joint friction in the URDF file as drake cannot handle it as of 17/06/2022 (version=??)
        try:
            linSys = Linearize(plant, context, input_port_index=input_i, output_port_index=output_i)
        except:
            print("Equilibrium check failed, setting equilibrium check tolerance to 1e-3 from 1e-6")
            linSys = Linearize(plant, context, input_port_index=input_i, output_port_index=output_i, equilibrium_check_tolerance=1e-3)

        A_cont, B_cont = linSys.A(), linSys.B()
        del builder, plant, context, input_i, output_i, scene_graph

        return A_cont, B_cont


def makePlantDiscrete(eq_state, eq_input, urdf, dt):
        """Makes plant of the system. Compatibility only tested for acrobot

        Args:
            eq_state (np.ndarray): System equilibrium point (states)
            eq_input (np.ndarray): System equilibrium point (input)
            urdf (string): Path to urdf file

        Returns:
            np.ndarray: A and B matrices of the linearized system
        """
        
        # Initiate diagram and make multibody plant using Acrobot.urdf
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=dt)  # Add a scene graph to the diagram
        Parser(plant).AddModelFromFile(urdf)  # Parse robot URDF file
        plant.Finalize()  # Finalize the acrobot plant

        # Set context
        context = plant.CreateDefaultContext()
        context.get_mutable_discrete_state_vector().SetFromVector(eq_state)
        plant.get_actuation_input_port().FixValue(context, eq_input)

        # Linearize the system to get the continuous A and B matrices
        input_i = plant.get_actuation_input_port().get_index()
        output_i = plant.get_state_output_port().get_index()

        # Relaxed equilibrium check tolerance was necessary for DFKI acrobot (default=1e-6)
        # Must comment out joint friction in the URDF file as drake cannot handle it as of 17/06/2022 (version=??)
        try:
            linSys = Linearize(plant, context, input_port_index=input_i, output_port_index=output_i)
        except:
            print("Equilibrium check failed, setting equilibrium check tolerance to 1e-3 from 1e-6")
            linSys = Linearize(plant, context, input_port_index=input_i, output_port_index=output_i, equilibrium_check_tolerance=1e-3)

        A_cont, B_cont = linSys.A(), linSys.B()
        del builder, plant, context, input_i, output_i, scene_graph

        return A_cont, B_cont


def controllability_matrix(A, B):
        """Generates the controllability matrix given the A and B matrices of a system
        C = [B, AB, A^2B, ..., A^(n-1)B]

        Args:
            A (np.ndarray): A matrix of a general state space system
            B (np.ndarray): B matrix of a general state space system

        Returns:
            np.ndarray: Controllability matrix
        """

        A_control = np.copy(A)
        B_control = np.copy(B)

        # Initiate the controllability matrix
        control = np.tile(np.zeros((len(A), len(B[0, :]))), (1, len(A) - 1))
        control = np.hstack((B_control, control))

        # Insert columns 1 by 1
        for i in range(len(A) - 1):
            B_control = A_control @ B_control
            control[:, len(B[0, :])*(i+1):len(B[0, :])*(i+1) + len(B[0, :])] = B_control
        
        del A_control, B_control
        return control


def controllabilityOverFreq(f_min, f_max, eq_state, eq_input, urdf, Q, R, multiplier=1):
    """This function calculates the controllability and eigenvalues of a discrete system over different values for the input frequency.

    Args:
        f_min (integer): Minimum frequency
        f_max (integer): Maximum frequency
        eq_state (np.ndarray): System equilibrium point (states)
        eq_input (np.ndarray): System equilibrium point (input)
        urdf (string): Path to urdf file
        Q (np.ndarray): Q matrix for LQR
        R (np.ndarray): R matrix for LQR
        multiplier (int, optional): Number of points tested within 1Hz of frequency. Defaults to 1.
    """

    # Generate array of frequencies and dt
    frequencies = np.linspace(f_min, f_max, int(multiplier*(f_max - f_min)) + 1, endpoint=True)
    delta_t = 1 / frequencies
    
    # Initialize dict of controllability Matrices, array of ranks, and array of eigevalues
    controlMatrices = {}
    ranks = np.zeros((int(multiplier*(f_max - f_min))  + 1, 1))
    eigenvalues = np.zeros((int(multiplier*(f_max - f_min)) + 1, len(eq_state) + 1))

    # Make plant
    A, B = makePlant(eq_state, eq_input, urdf)
    
    # Iterate over all combinations of dt
    for time_step_iter, dt in tqdm(enumerate(delta_t)):
        
        # Calculate discrete time system A and B matrix
        Ad = expm(dt*(A))
        Bd = np.linalg.inv(A) @ (Ad - np.eye(len(A))) @ B

        # Calculate the discrete LQR gain matrix
        K = DiscreteTimeLinearQuadraticRegulator(Ad, Bd, Q, R)[0]

        # Get controllability matrix, rank, and eigenvalues
        eigvalsControlMat = np.linalg.eigvals(Ad-Bd@K)
        controlMat = controllability_matrix(Ad, Bd)
        rankControlMat = np.linalg.matrix_rank(controlMat)

        # Append to dict/array
        controlMatrices[time_step_iter] = controlMat
        ranks[time_step_iter] = rankControlMat
        eigenvalues[time_step_iter, :] = np.hstack((np.array(dt), np.abs(eigvalsControlMat)))

        del Ad, Bd, K, eigvalsControlMat, controlMat, rankControlMat

    return controlMatrices, ranks, eigenvalues, frequencies, delta_t, A, B


def SimWithRate(A, B, C, D, Q, R, f0, x0, eq_state, eq_input):
    """Runs a simulation of the discrete system at 500Hz. Compatibility only tested with acrobot

    Args:
        A (np.ndarray): A matrix of a continuous state space system
        B (np.ndarray): B matrix of a continuous state space system
        C (np.ndarray): C matrix of a discrete state space system
        D (np.ndarray): D matrix of a discrete state space system
        f0 (float): update frequency in Hz
        x0 (np.ndarray): Initial states of the system

    Returns:
        np.ndarray: Log of the states
    """

    # Make a plant and discrete LQR controller
    builder = DiagramBuilder()
    dt = 1/f0
    # Ad = expm(dt*(A))
    # Bd = np.linalg.inv(A) @ (Ad - np.eye(len(A)))@B
    Ad = A
    Bd = B
    Kd = DiscreteTimeLinearQuadraticRegulator(Ad, Bd, Q, R)[0]

    plant = LinearSystem(Ad, Bd, C, D, dt)
    context = plant.CreateDefaultContext()
    context.get_mutable_discrete_state_vector().SetFromVector(eq_state)
    
    plant.get_input_port().FixValue(context, eq_input)
    plant = builder.AddSystem(plant)
    lqr = builder.AddSystem(MatrixGain(-Kd))

    # Input saturation limits and reference to be tracked 
    saturation = builder.AddSystem(Saturation(min_value=[-3000], max_value=[3000]))
    adder = builder.AddSystem(Adder(2, len(Ad)))
    vector = builder.AddSystem(ConstantVectorSource(-eq_state))
    
    # Wrapping angles for q0 and q1
    wrapangles = WrapToSystem(4)
    wrapangles.set_interval(0, 0, 2. * np.pi)
    wrapangles.set_interval(1, -np.pi, np.pi)
    wrapto = builder.AddSystem(wrapangles)

    # Zero Order hold
    ZOH = builder.AddSystem(ZeroOrderHold(period_sec=dt, vector_size=np.shape(Bd)[1]))

    # Connect individual elements of diagrams
    builder.Connect(saturation.get_output_port(), ZOH.get_input_port())
    builder.Connect(ZOH.get_output_port(), plant.get_input_port())
    builder.Connect(plant.get_output_port(), wrapto.get_input_port())
    builder.Connect(wrapto.get_output_port(), adder.get_input_port(0))
    builder.Connect(vector.get_output_port(), adder.get_input_port(1))
    builder.Connect(adder.get_output_port(), lqr.get_input_port())
    builder.Connect(lqr.get_output_port(), saturation.get_input_port())

    # Connect the output log to the plant's state output
    logger_state = LogVectorOutput(plant.get_output_port(), builder)
    
    # Finalize diagram and visualize it
    diagram = builder.Build()
    diagram.set_name("diagram")

    display(SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].create_svg()))

    # Initiate simulator
    simulator = Simulator(diagram)
    # ResetIntegratorFromFlags(simulator=simulator, scheme="runge_kutta2", max_step_size=1/(5*1000.))

    # reset initial time and state
    context = simulator.get_mutable_context()
    context.SetTime(0.)
    context.SetDiscreteState(0, x0)

    # run sim for 10 seconds
    simulator.Initialize()
    simulator.AdvanceTo(10)

    return logger_state.FindLog(simulator.get_context()).data()


if __name__ == "__main__": 
    
    import seaborn as sns
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "sans-serif"
    sns.set_palette("dark")
    sns.set_context("talk")

    Q = np.diag((10., 10., 1., 1.))
    R = np.array([1.])

    eq_state = np.array([np.pi, 0., 0., 0.])
    eq_input = 0.

    C = np.eye(4)
    D = np.zeros((4, 1))

    matrices, ranks, eigenvalues, freq, delta, Ad_tmotor, Bd_tmotor = controllabilityOverFreq(50, 1000, eq_state, eq_input, "Acrobot/acrobot_tmotors.urdf", Q, R)
    matrices2, ranks2, eigenvalues2, freq2, delta2, Ad_mj, Bd_mj = controllabilityOverFreq(50, 1000, eq_state, eq_input, "Acrobot/acrobot_mjbots.urdf", Q, R)

    freq = np.flip(freq)
    freq2 = np.flip(freq2)

    show=False

    # print((eigenvalues[:5, :] - eigenvalues2[:5, :]))
    # print(eigenvalues[:5, :])
    # print(eigenvalues2[:5, :])

    plt.plot(1/eigenvalues[:, 0], eigenvalues[:, 1], label=r"$\lambda_{1}$")
    plt.plot(1/eigenvalues[:, 0], eigenvalues[:, 2], label=r"$\lambda_{2}$")
    plt.plot(1/eigenvalues[:, 0], eigenvalues[:, 3], label=r"$\lambda_{3}$")
    plt.plot(1/eigenvalues[:, 0], eigenvalues[:, 4], label=r"$\lambda_{4}$")
    plt.title("Eigenvalues discrete Acrobot TMotors")
    plt.xlabel("frequency [Hz]")
    plt.ylabel(r"$|\lambda|$")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
        pass
    else:
        plt.savefig("Figures/discrete_acrobot_tmotors.png", dpi=400)
        plt.close()


    plt.plot(1/eigenvalues2[:, 0], eigenvalues2[:, 1], label=r"$\lambda_{1}$")
    plt.plot(1/eigenvalues2[:, 0], eigenvalues2[:, 2], label=r"$\lambda_{2}$")
    plt.plot(1/eigenvalues2[:, 0], eigenvalues2[:, 3], label=r"$\lambda_{3}$")
    plt.plot(1/eigenvalues2[:, 0], eigenvalues2[:, 4], label=r"$\lambda_{4}$")
    plt.title("Eigenvalues discrete acrobot MJbots")
    plt.xlabel("frequency [Hz]")
    plt.ylabel(r"$|\lambda|$")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig("Figures/discrete_acrobot_mjbots.png", dpi=400)
        plt.close()

    # Simulation of the discrete system
    A_disc, B_disc = makePlantDiscrete(eq_state, eq_input, "Acrobot/acrobot_tmotors.urdf", 1/500.)
    sim_tmotor = SimWithRate(A_disc, B_disc, C, D, Q, R, 500., eq_state+0.01*np.random.randn(4), eq_state, eq_input)
    t_sim_tmotor = np.linspace(0, 10, np.shape(sim_tmotor)[1])

    plt.plot(t_sim_tmotor, sim_tmotor[0, :])
    # plt.ylim([-7, 7])
    plt.show()

    # sim_mj = SimWithRate(Ad_mj, Bd_mj, C, D, Q, R, 500., np.array([np.pi, 0., 0., 0.]), eq_state, eq_input)
    # t_sim_mj = np.linspace(0, 10, np.shape(sim_mj)[1])

    # plt.plot(t_sim_tmotor, sim_tmotor[0, :])
    # # plt.ylim([-7, 7])
    # plt.show()

