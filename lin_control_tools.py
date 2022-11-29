"""This script servers the purpose of making """

import numpy as np


def controllability_matrix(A, B):
        """Generates the controllability matrix given the A and B matrices of a system
        R = [B, AB, A^2B, ..., A^(n-1)B]

        Args:
            A (np.ndarray): A (nxn) matrix of a general state space system
            B (np.ndarray): B (nxm) matrix of a general state space system

        Returns:
            np.ndarray: Controllability matrix
        """

        # Make copy of A and B matrix
        A_control = np.copy(A)
        B_control = np.copy(B)

        # Initiate the controllability matrix
        control = np.tile(B_control, (1, len(A)))

        # Insert A^(i)B blocks iteratively
        for i in range(len(A) - 1):
            B_control = A_control @ B_control
            control[:, len(B[0, :])*(i+1):len(B[0, :])*(i+2)] = B_control
        
        return control


def observability_matrix(A, C):
    """Generates the observability matrix given the A and C matrices of a system
        W = [C,
             CA,
             ...,
             CA^(n-1)]

        Args:
            A (np.ndarray): A (nxn) matrix of a general state space system
            C (np.ndarray): C (pxn) matrix of a general state space system

        Returns:
            np.ndarray: Observability matrix (npxn)
        """

    # Make copy of A and C matrix
    A_observe = np.copy(A)
    C_observe = np.copy(C)

    # Initiate the controllability matrix
    observe = np.tile(C_observe, (len(A), 1))

    # Insert CA^(i) blocks iteratively
    for i in range(len(A) - 1):
        C_observe = C_observe @ A_observe
        observe[len(C)*(i+1):len(C)*(i+2), :] = C_observe

    return observe


def controllability(A, B):
    control_mat = controllability_matrix(A, B)

    if np.linalg.matrix_rank(control_mat) == len(A):
        return True
    else:
        return False


def observability(A, C):
    observe_mat = observability_matrix(A, C)

    if np.linalg.matrix_rank(observe_mat) == len(A):
        return True
    else:
        return False


def stabilizability(A, B, K, sys_type):
    """ Determines whether a system is stabilizable for a given gain matrix
        
        Args:
            A (np.ndarray): A (nxn) matrix of a general state space system
            B (np.ndarray): B (nxm) matrix of a general state space system
            K (np.ndarray): K (mxn) gain matrix
            sys_type (str): 'continuous' or 'discrete' 

        Returns:
            Boolean: is the system stabilizable
    """

    # First check for stricter condition of controllability
    if controllability(A, B):
        return True
    else:
        sys = A - B @ K
        eigvals_sys = np.linalg.eigvals(sys)

        if sys_type.lower().startswith("c"):
            if eigvals_sys[np.real(eigvals_sys) > 0].size == 0:
                return True
            else:
                return False
        elif sys_type.lower().startswith("d"):
            if eigvals_sys[np.abs(np.real(eigvals_sys)) < 1].size == 0:
                return True
            else:
                return False
        else:
            raise TypeError(f"sys_type must be 'continuous' or 'discrete', received {sys_type}")
            

def detectability(A, C, K, sys_type):
    """ Determines whether a system is stabilizable for a given gain matrix
        
        Args:
            A (np.ndarray): A (nxn) matrix of a general state space system
            C (np.ndarray): C (pxn) matrix of a general state space system
            K (np.ndarray): K (nxp) gain matrix
            sys_type (str): 'continuous' or 'discrete' 

        Returns:
            Boolean: is the system stabilizable
    """

    # First check if system is observable as that is an easy to compute stricter criterion
    if observability(A, C):
        return True
    else:
        raise NotImplementedError("The full detectability method hasn't been implemented yet")


def kalman_decomposition(A, B, C):

    control_mat = controllability_matrix(A, B)
    observe_mat = observability_matrix(A, C)

    raise NotImplementedError("Kalman Decomposition function is not fully implemented yet")


if __name__ == "__main__":
    
    # Making system of equations (A, B, C)

    A = np.random.randn(20, 20) # 20x20 A matrix
    B = np.ones((20, 3))        # 20x3 B matrix

    C = np.eye(20)              # Identity matrix 
    
    # Test shape observability matrix
    observe_mat = observability_matrix(A, C)
    print(np.shape(observe_mat))

    # Test shape controllability matrix
    control_mat = controllability_matrix(A, B)
    print(np.shape(control_mat))