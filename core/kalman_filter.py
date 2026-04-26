import numpy as np


class KalmanFilter:
    """
    4-state Kalman filter: [x, y, vx, vy].
    Measurement is center position [x, y].
    """

    def __init__(self) -> None:
        # State transition: x_{k} = F * x_{k-1}
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Observation: z = H * x
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.state = np.zeros((4, 1), dtype=np.float32)
        self.P     = np.eye(4, dtype=np.float32) * 100.0
        self.Q     = np.eye(4, dtype=np.float32)       # process noise
        self.R     = np.eye(2, dtype=np.float32) * 6.0 # measurement noise

    def init_state(self, cx: float, cy: float) -> None:
        self.state = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.P     = np.eye(4, dtype=np.float32) * 100.0

    def set_process_noise(self, speed: float) -> None:
        q_pos = float(np.clip(1.0 + speed * 0.5, 1.0, 12.0))
        q_vel = float(np.clip(8.0 + speed * 0.5, 8.0, 30.0))
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)

    def set_measurement_noise(self, r: float) -> None:
        self.R = np.eye(2, dtype=np.float32) * max(0.1, r)

    def predict(self) -> tuple[float, float]:
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q
        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        z = np.array([[cx], [cy]], dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P     = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        return float(self.state[0, 0]), float(self.state[1, 0])
