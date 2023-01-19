import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from settings import *

G = 1.0
CENTER_MASS = 100000.0
BODY_PERIOD_LO = 100.0
BODY_PERIOD_HI = 1000.0
ORBIT_R_LO = 10.0
ORBIT_R_HI = 100.0
DT = 0.01
TIME_STEPS = 5000
N_BODIES = 40
SOFTENING_SQ = 0.01


class Bodies:
    def __init__(self, m, pos, velocity):
        """
        Body initialization
        :param m: Body mass
        :param pos: Body position vector (x, y) in world units
        :param velocity: Body velocity (vx, vy)
        """
        self.m = m
        self.pos = pos
        self.velocity = velocity
        self.acceleration = np.zeros((m.shape[0], 3))

    def update(self):
        mass = self.m

        dt = DT

        self.velocity += self.acceleration * dt * 0.5
        self.pos += self.velocity * dt

        x = self.pos[:, 0:1]
        y = self.pos[:, 1:2]
        z = self.pos[:, 2:3]

        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz
        r2 = dx2 + dy2 + dz2
        inv_r3 = (r2 + SOFTENING_SQ)**(-1.5)

        ax = G * (dx * inv_r3) @ mass
        ay = G * (dy * inv_r3) @ mass
        az = G * (dz * inv_r3) @ mass

        self.acceleration = np.stack((ax, ay, az), axis=1)
        self.velocity += self.acceleration * dt * 0.5


def get_next_time_step_pos(bodies):
    bodies.update()
    return bodies.pos


def update_lines(num, body_trajectories, plt_lines):
    for line, data in zip(plt_lines, body_trajectories):
        line.set_data(data[0:2, max(0, num-200):num])  # Trajectory for the body until time-step num
        line.set_3d_properties(data[2, max(0, num-200):num])  # Z-coordinates for the body until time-step num
        line.set(alpha=0.5)
    return plt_lines


def get_rotation_matrix(angle):
    cx = np.cos(angle[0])
    sx = np.sin(angle[0])
    cy = np.cos(angle[1])
    sy = np.sin(angle[1])
    cz = np.cos(angle[2])
    sz = np.sin(angle[2])

    rot_x = np.array([[1.0, 0.0, 0.0],
                      [0.0, cx, sx],
                      [0.0, -sx, cx]])
    rot_y = np.array([[cy, 0.0, -sy],
                      [0.0, 1.0, 0.0],
                      [sy, 0.0, cy]])
    rot_z = np.array([[cz, sz, 0.0],
                      [-sz, cz, 0.0],
                      [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def generate_body(orbit_r, orbit_angle):
    period = np.random.uniform(BODY_PERIOD_LO, BODY_PERIOD_HI)
    m_exp = 4.0 * np.pi * np.pi * orbit_r**3 / (G * period * period)
    m_sigma = m_exp / 3.0
    m = np.random.normal(m_exp, m_sigma)
    v_exp = np.sqrt(G * CENTER_MASS / orbit_r)
    v_sigma = v_exp / 3.0
    v = np.random.normal(v_exp, v_sigma)
    velocity = orbit_angle @ np.asarray([0.0, v, 0.0])
    pos = orbit_angle @ np.asarray([orbit_r, 0.0, 0.0])
    return m, pos, velocity


def main():
    # Fixing random state for reproducibility
    np.random.seed(777)

    m = np.empty(N_BODIES)
    pos = np.empty((N_BODIES, 3))
    vel = np.empty((N_BODIES, 3))

    m[0] = CENTER_MASS
    pos[0] = np.zeros(3)
    vel[0] = np.zeros(3)

    for i in range(1, N_BODIES):
        orbit_radius = np.random.uniform(ORBIT_R_LO, ORBIT_R_HI)
        orbit_angle = np.asarray([0.0, 0.0, np.random.uniform(0.0, 6.0)])
        rot_matrix = get_rotation_matrix(orbit_angle)
        bm, bp, bv = generate_body(orbit_radius, rot_matrix)
        m[i] = bm
        pos[i] = bp
        vel[i] = bv
    bodies = Bodies(m, pos, vel)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Generate trajectories
    trajectories = np.empty((N_BODIES, 3, TIME_STEPS))
    for time_step in range(TIME_STEPS):
        trajectories[..., time_step] = get_next_time_step_pos(bodies)

    # For each trajectory creating respective axis line object initialized with initial point only
    axis_lines = [ax.plot(trajectories[i, 0, 0:1],
                          trajectories[i, 1, 0:1],
                          trajectories[i, 2, 0:1])[0] for i in range(trajectories.shape[0])]

    # Setting the axes properties
    ax.set_xlim3d([-ORBIT_R_HI, ORBIT_R_HI])
    ax.set_xlabel('X')

    ax.set_ylim3d([-ORBIT_R_HI, ORBIT_R_HI])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-ORBIT_R_HI, ORBIT_R_HI])
    ax.set_zlabel('Z')

    ax.set_title('N-Body Simulation')

    # Creating the Animation object
    lines_anim = animation.FuncAnimation(fig, update_lines, TIME_STEPS,
                                         fargs=(trajectories, axis_lines), interval=1, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
