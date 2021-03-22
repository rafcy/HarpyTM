import numpy as np
from numpy.linalg import inv

# Kalman Filter Class
class KalmanFilter:
    """
    Simple Kalman filter
    """

    def __init__(self, XY, B=np.array([0]), M=np.array([0])):
        stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
        if XY != 0:
            stateMatrix = np.array([[XY[0]], [XY[1]], [0], [0]], np.float32)
            # np.array([XY[0], XY[1],0,0],np.float32)
        estimateCovariance = np.eye(stateMatrix.shape[0])
        transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.001
        measurementStateMatrix = np.zeros((2, 1), np.float32)
        observationMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 13
        X=stateMatrix
        P=estimateCovariance
        F=transitionMatrix
        Q=processNoiseCov
        Z=measurementStateMatrix
        H=observationMatrix
        R=measurementNoiseCov
        """
        Initialise the filter
        Args:
            X: State estimate
            P: Estimate covariance
            F: State transition model
            B: Control matrix
            M: Control vector
            Q: Process noise covariance
            Z: Measurement of the state X
            H: Observation model
            R: Observation noise covariance
        """
        self.X = X
        self.P = P
        self.F = F
        self.B = B
        self.M = M
        self.Q = Q
        self.Z = Z
        self.H = H
        self.R = R

    def predict(self):
        """
        Predict the future state
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            self.B: Control matrix
            self.M: Control vector
        Returns:
            updated self.X
        """
        # Project the state ahead
        self.X = self.F @ self.X + self.B @ self.M
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X

    def correct(self, Z):
        """
        Update the Kalman Filter from a measurement
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            Z: State measurement
        Returns:
            updated X
        """
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)
        self.X += K @ (Z - self.H @ self.X)
        self.P = self.P - K @ self.H @ self.P

        return self.X

"""
needed variables to instantly initialize kalman with just parsing X,Y variables (x,y) 
"""
def init_kalman(XY):
    kalman = None
    stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
    if XY != 0:
        stateMatrix = np.array([[XY[0]], [XY[1]],[0],[0]],np.float32)
        # np.array([XY[0], XY[1],0,0],np.float32)
    estimateCovariance = np.eye(stateMatrix.shape[0])
    transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.001
    measurementStateMatrix = np.zeros((2, 1), np.float32)
    observationMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 13
    kalman = KalmanFilter(X=stateMatrix,
                          P=estimateCovariance,
                          F=transitionMatrix,
                          Q=processNoiseCov,
                          Z=measurementStateMatrix,
                          H=observationMatrix,
                          R=measurementNoiseCov)
    return kalman
