import numpy as np
from pydrake.all import (BasicVector, Cylinder, GeometryInstance, RotationMatrix, RigidTransform,
                        AbstractValue, StartMeshcat, MakePhongIllustrationProperties)    

from pydrake.systems.framework import LeafSystem, EventStatus
from pydrake.multibody import inverse_kinematics
from pydrake.multibody import plant as plnt
from pydrake.multibody import math as m

from pydrake.multibody.plant import ContactResults
from underactuated.meshcat_cpp_utils import MeshcatSliders
import pydrake.multibody.meshcat as meshcat
import pydrake.geometry as g
import meshcat.transformations as tf

# Reads the Abstract Value contact results from the contact_results port and sends the contact force of the first contact point pair
# As well as telling whether the system is in contact with an object
class ReadContactResults(LeafSystem):

    def __init__(self, plant, links):

        self.plant = plant
        self.links = links
        
        LeafSystem.__init__(self)
        
        # Input port for the contact_results port of a multibody plant
        self.inputport = self.DeclareAbstractInputPort("contact_results_input", AbstractValue.Make(ContactResults()))
        
        # Output ports of this system, port 1 gives the contact forces of the first contact pair
        # port 2 returns whether the system is in contact with a surface (0 or 1)
        self.DeclareVectorOutputPort("contact_force", BasicVector(3), self.DoCalcVectorOutput)
        self.DeclareVectorOutputPort("is_contact", BasicVector(1), self.IsContact)

    def DoCalcVectorOutput(self, context, output):
        
        # get the number of point pair contacts of the system
        num_contacts = self.inputport.Eval(context).num_point_pair_contacts()

        # If there is a contact get the contact force, otherwise return a zero vector
        if num_contacts > 0:
            contact_forces = self.inputport.Eval(context).point_pair_contact_info(0).contact_force()
        else:
            contact_forces = np.zeros(3)
        output.SetFromVector(contact_forces)

        self.out = output
    
    def IsContact(self, context, output):
        
        # link_index = []
        # for points in self.links:
        #     link_index.append(self.plant.GetBodyByName(points).index())

        num_conacts = self.inputport.Eval(context).num_point_pair_contacts()

        if num_conacts  == len(self.links):
            contacting = np.array([1])
        else:
            contacting= np.array([0])

        output.SetFromVector(contacting)


# Switches controller based on slider value
# Currently no hopping controller implemented 
class SwitchController(LeafSystem):

    def __init__(self):
        LeafSystem.__init__(self)
        self.balance = self.DeclareVectorInputPort("balance_torque", BasicVector(2))
        self.hopping = self.DeclareVectorInputPort("hopping_torque", BasicVector(2))
        self.button = self.DeclareVectorInputPort("button_input", BasicVector(1))

        self.torque_out = self.DeclareVectorOutputPort("torque_output", BasicVector(2), self.DoCalcVectorOutput)

    def DoCalcVectorOutput(self, context, output):
        
        button_status = self.button.Eval(context)

        if button_status == 0:
            torque = self.balance.Eval(context)
        else:
            torque = self.hopping.Eval(context)
        
        output.SetFromVector(torque)


# Balance controller for the leg
# Might be extended to hopping 
class Balancing(LeafSystem):
    def __init__(self, lqr_gain):

        self.K = lqr_gain

        LeafSystem.__init__(self)
        self.cont_state_input = self.DeclareVectorInputPort("continuous_state", BasicVector(6))
        self.desired_state_input = self.DeclareVectorInputPort("desired_state", BasicVector(8))
        self.contact = self.DeclareVectorInputPort("is_contact", BasicVector(1))

        self.output_balance = self.DeclareVectorOutputPort("balancing_output", BasicVector(2), self.DoBalance)
        self.output_hopping = self.DeclareVectorOutputPort("hopping_output", BasicVector(2), self.DoHopping)
        self.near_equilibrium = False

    def DoBalance(self, context, output):

        state = self.cont_state_input.Eval(context)
        desired_state = self.desired_state_input.Eval(context)

        Kp_hip, Kd_hip = 25., 5.
        Kp_knee, Kd_knee = 25., 5.

        if self.contact.Eval(context) == 1:

            # PD controller
            # torque_hip = -Kp_hip * (state[1] - desired_state[1]) - Kd_hip * (state[4] - desired_state[4])
            # torque_knee = -Kp_knee * (state[2] - desired_state[2]) - Kd_knee * (state[5] - desired_state[5])
            # torque = np.array([torque_hip, torque_knee])

            # LQR controller with manual output saturation 
            # (doesn't properly cap torque somehow, just like the saturation block)
            torque = -self.K@(state - desired_state[:6])
            torque[torque < -6] = -6
            torque[torque > 6] = 6

            output.SetFromVector(torque)

        else:

            torque = np.zeros(2)
            output.SetFromVector(torque)

    def DoHopping(self, context, output):
        # TODO (low priority) Implement energy shaping to set torques
        test = self.contact.Eval(context)
        output.SetFromVector(np.zeros(2))


# Apply a disturbance force at some time during the simulation
class Disturbance(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.contact = self.DeclareVectorInputPort("is_contact", BasicVector(1))
        self.force = self.DeclareAbstractOutputPort("spatial_force", lambda: AbstractValue.Make([plnt.ExternallyAppliedSpatialForce()]), self.CalcDisturb)
        self.vectorLogging = self.DeclareVectorOutputPort("force log", BasicVector(3), self.DoLog)

        self.force_object = plnt.ExternallyAppliedSpatialForce()
        self.force_object.body_index = plant.GetBodyByName("Link_Knee_Pitch").index()
        self.force_object.p_BoBq_B = np.zeros(3) + np.array([0.03, -0.05, 0.01])

    def CalcDisturb(self, context, output):

        if self.contact.Eval(context) == 1 and 1 <= context.get_time() <= 1.05:

            # self.rand_force = 2*np.random.randn(3, 1)
            self.rand_force = -200*np.ones(3)  # X --> sideways, Y --> nothing, Z --> nothing
            self.rand_tau = 0.2*np.zeros((3, 1))
            self.force_object.F_Bq_W = m.SpatialForce(tau=self.rand_tau, f=self.rand_force)
            output.set_value([self.force_object])
            
        else:
            
            self.rand_force = np.zeros((3, 1))
            self.rand_tau = np.zeros((3, 1))
            self.force_object.F_Bq_W = m.SpatialForce(tau=self.rand_tau, f=self.rand_force)
            output.set_value([self.force_object])

    
    def DoLog(self, output):
        output.SetFromVector(self.rand_force)


class VisualizeForce(LeafSystem):

    def __init__(self, plant, visualizer):
        LeafSystem.__init__(self)
        self.DeclarePeriodicUnrestrictedUpdateEvent(1/100, 0.0, update=self.DrawFrame)
        self.vis = meshcat.Visualizer()
        self.force_input = self.DeclareAbstractInputPort("force_in", AbstractValue.Make([plnt.ExternallyAppliedSpatialForce()]))
        self.dummy_out = self.DeclareVectorOutputPort("dummy_out", BasicVector(1), self.DrawFrame)
        self.plant = plant

    def DrawFrame(self, context, output):
        
        print("Published")

        force_object = self.force_input.Eval(context)
        
        WF = self.plant.world_frame()
        BF = self.plant.GetBodyByName("Link_Knee_Pitch").body_frame()

        frame = self.vis["frame"]
        x_axis = frame["x_axis"]
        y_axis = frame["y_axis"]
        z_axis = frame["z_axis"]

        # Make axes with force magnitudes
        forces = force_object[0].F_Bq_W.get_coeffs()[3:]
        
        # Normalize forces to ~0.1m

        x_axis["cyl"].set_object(g.Cylinder(height=np.abs(forces[0]), radius=0.005))
        y_axis["cyl"].set_object(g.Cylinder(height=np.abs(forces[1]), radius=0.005))
        z_axis["cyl"].set_object(g.Cylinder(height=np.abs(forces[2]), radius=0.005))

        x_axis["cone"].set_object(g.Cylinder(height=np.abs(forces[0])/10, radiusTop=0, radiusBottom=0.01))
        y_axis["cone"].set_object(g.Cylinder(height=np.abs(forces[1])/10, radiusTop=0, radiusBottom=0.01))
        z_axis["cone"].set_object(g.Cylinder(height=np.abs(forces[2])/10, radiusTop=0, radiusBottom=0.01))

        # Replace rotation matrices with Drake rotations

        x_axis["cyl"].set_transform(tf.rotation_matrix(-np.pi/2, [0, 0, 1]))
        z_axis["cyl"].set_transform(tf.rotation_matrix(np.pi/2, [1, 0, 0]))

        x_axis["cone"].set_transform(tf.rotation_matrix(-np.pi/2, [0, 0, 1]))
        z_axis["cone"].set_transform(tf.rotation_matrix(np.pi/2, [1, 0, 0]))

        x_axis.set_transform(tf.translation_matrix([np.abs(forces[0]) / 2, 0, 0]))
        y_axis.set_transform(tf.translation_matrix([0, np.abs(forces[1]) / 2, 0]))
        z_axis.set_transform(tf.translation_matrix([0, 0, np.abs(forces[2]) / 2]))

        x_axis["cone"].set_transform(tf.translation_matrix([np.abs(forces[0]) / 2, 0, 0]))
        y_axis["cone"].set_transform(tf.translation_matrix([0, np.abs(forces[1]) / 2, 0]))
        z_axis["cone"].set_transform(tf.translation_matrix([0, 0, np.abs(forces[2]) / 2]))

        
class LQR_system(LeafSystem):
    def __init__(self, plant, nr_states, nr_outputs, LQRgains):
        LeafSystem.__init__(self)

        self.K = LQRgains
        self.counter = 0

        self.state = self.DeclareVectorInputPort("continuous_state", BasicVector(nr_states))
        self.desired_states = self.DeclareVectorInputPort("desired_state", BasicVector(nr_states))
        self.tau_0 = self.DeclareVectorInputPort("tau_0", BasicVector(plant.num_actuators()))

        self.tau_out = self.DeclareStateOutputPort("torques_output", self.DeclareDiscreteState(nr_outputs))
        self.DeclarePeriodicDiscreteUpdateEvent(1/50., 0.0, self.DoCalcVectorOutput)

    def DoCalcVectorOutput(self, context, output):
        states = self.state.Eval(context)
        desired_state = self.desired_states.Eval(context)
        tau_0 = self.tau_0.Eval(context)

        input = tau_0 - self.K @ (states - desired_state)

        input[input > 10] = 10
        input[input < -10] = -10

        # print(f"Update {self.counter} at time {context.get_time()}, Torque supplied: {input}", end="\r")
        # self.counter += 1

        # if states[0] < -np.pi/2 or states[0] > np.pi/2 or self.counter > 250000:
        #     self.counter = 0
        #     raise TimeoutError

        output.set_value(input)


class contact_LQR(LeafSystem):
    def __init__(self, nr_states, nr_outputs, LQRgains, x0, tau_0):
        LeafSystem.__init__(self)

        self.K = LQRgains
        self.counter = 0
        self.tau_0 = tau_0
        self.x0 = x0

        self.nr_outputs = nr_outputs

        self.state = self.DeclareVectorInputPort("continuous_state", BasicVector(nr_states))
        self.contact = self.DeclareVectorInputPort("is_contact", BasicVector(1))
        
        self.output_port = self.DeclareVectorOutputPort("system_inputs", BasicVector(nr_outputs), self.DoCalcVectorOutput)

    def DoCalcVectorOutput(self, context, output):
        
        print(context.get_time(), end="\r")
        states = self.state.Eval(context)
        is_contact = self.contact.Eval(context)

        if is_contact == 0:
            input = np.zeros(self.nr_outputs)
        
        else:
            input = self.tau_0 - self.K @ (states[1:] - self.x0[1:])    # Ignore q0 for floating base systems

            input[input > 50] = 50
            input[input < -50] = -50

            # print(f"Update {self.counter} at time {context.get_time()}, Torque supplied: {input}", end="\r")
            # self.counter += 1

            # if states[0] < -np.pi/2 or states[0] > np.pi/2 or self.counter > 250000:
            #     self.counter = 0
            #     raise TimeoutError

        output.SetFromVector(input)


class PD_system(LeafSystem):
    def __init__(self, nr_states, nr_outputs, plant):
        LeafSystem.__init__(self)

        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self.CoM_y = self._plant.CalcCenterOfMassPositionInWorld(self._plant_context)[1]

        self.Kp = 25*np.array([1, 1, 1, 1])
        self.Kd = 5*np.array([1, 1, 1, 1])

        self.state = self.DeclareVectorInputPort("continuous_state", BasicVector(nr_states))
        self.desired_states = self.DeclareVectorInputPort("desired_state", BasicVector(nr_states))
        self.output_port = self.DeclareVectorOutputPort("system_inputs", BasicVector(nr_outputs), self.DoCalcVectorOutput)
        self.CoM_log = self.DeclareVectorOutputPort("CoM_pos_y", BasicVector(1), self.LogCoM)

        self.ff_tau = np.zeros(4)

    def DoCalcVectorOutput(self, context, output):

        states = self.state.Eval(context)
        desired_states = self.desired_states.Eval(context)

        self._plant.SetPositions(self._plant_context, states[:5])
        self.CoM_y = self._plant.CalcCenterOfMassPositionInWorld(self._plant_context)[1]

        states[4] = self.CoM_y

        input = -self.Kp * (states[1:5] - desired_states[1:5]) - self.Kd * (states[6:] - desired_states[6:]) + self.ff_tau

        output.SetFromVector(input)
    
    def LogCoM(self, context, output):
        output.SetFromVector([self.CoM_y])


if __name__ == "__main__":
    pass