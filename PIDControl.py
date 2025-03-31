import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0
    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error  
        return output
    def auto_tune(self, error, prev_error, dt):
        change_error = error - prev_error
        
        if abs(change_error) > 0.1:  
            self.kp += 0.001
        if abs(change_error) > 0.5:  
            self.kd += 1e-5
        if abs(error) < 0.5 and abs(self.integral) > 1:  #
            self.ki += 1e-5
def move_bot(start, path_points, kp=0.0, ki=0.0, kd=0.0, dt=0.1):
    x, y = start
    path = [(x, y)]
    theta = 0.0 
    pid_bot = PIDController(kp, ki, kd)
    prev_error = 0
    for point in path_points:
        while np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) > 0.1:
            error = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
            # Auto-tune PID
            pid_bot.auto_tune(error, prev_error, dt)
            prev_error = error  # Update prev_error
            control_signal = pid_bot.compute(error, dt)
            angle_deviation = np.arctan2((point[1] - y), (point[0] - x))
            angle_error = angle_deviation - theta
            theta += 0.5 * angle_error
            x += control_signal * np.cos(theta) * dt
            y += control_signal * np.sin(theta) * dt
            path.append((x, y))

    return path, pid_bot  

# Define start point and waypoints
start = (0, 0)
path_points = [(47, 3), (17, 62)]
kp, ki, kd = 0.0, 0.0, 0.0  
actual_path, tuned_pid = move_bot(start, path_points, kp, ki, kd)
actual_path = np.array(actual_path)

plt.plot(actual_path[:, 0], actual_path[:, 1], label="PID Path", color="blue")
ideal_x = [start[0]] + [p[0] for p in path_points]
ideal_y = [start[1]] + [p[1] for p in path_points]
plt.plot(ideal_x, ideal_y, 'k--', label="Ideal Path", linewidth=1.5)
plt.scatter(*zip(*path_points), color='red', label="Waypoints")
plt.scatter(*start, color='green', label="Start")

# Display auto-tuned PID values
pid_text = f"Kp = {tuned_pid.kp:.4f}\nKi = {tuned_pid.ki:.4f}\nKd = {tuned_pid.kd:.4f}"
plt.text(min(actual_path[:, 0]), max(actual_path[:, 1]), pid_text, fontsize=12, color="blue",
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Auto-Tuning PID Bot Path")
plt.grid()
plt.show()
