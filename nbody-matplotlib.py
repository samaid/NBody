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
STEPS_TO_PLOT = 200


class Bodies:
    def __init__(self, m, pos, velocity):
        """
        Body initialization
        :param m: Body mass
        :param pos: Body position (x, y, z)
        :param velocity: Body velocity (vx, vy, vz)
        """
        self.m = m
        self.pos = pos
        self.velocity = velocity
        self.acceleration = np.zeros((m.shape[0], 3))

    def update(self):
        """
        Updates accelerations, velocities and positions of each body
        :return: None
        """
        mass = self.m

        dt = DT

        self.velocity += self.acceleration * dt * 0.5  # Perform half-step with old acceleration
        self.pos += self.velocity * dt  # Update position with old velocity

        x = self.pos[:, 0:1]
        y = self.pos[:, 1:2]
        z = self.pos[:, 2:3]

        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz
        r2 = dx2 + dy2 + dz2  # Squared distance matrix
        inv_r3 = (r2 + SOFTENING_SQ)**(-1.5)  # Use softening for avoiding infinities when two objects are close

        ax = G * (dx * inv_r3) @ mass
        ay = G * (dy * inv_r3) @ mass
        az = G * (dz * inv_r3) @ mass

        self.acceleration = np.stack((ax, ay, az), axis=1)
        self.velocity += self.acceleration * dt * 0.5  # Perform half-step with new acceleration


def get_next_time_step_pos(bodies):
    """
    Perform update and return new position.

    This is utility function used for trajectories generation

    :param bodies: Bodies that interact each other
    :return: New position
    """
    bodies.update()
    return bodies.pos


def update_lines(time_step, body_trajectories, plt_lines):
    """
    Utility function that updates line objects for the 3D plot.

    For better visual experience we do not plot entire trajectories,
    rather only STEPS_TO_PLOT last time-steps

    :param time_step: Current time-step
    :param body_trajectories: Body trajectories with data to update 3D axis lines
    :param plt_lines: Axis lines to update
    :return: Updated 3d plot lines
    """
    for line, data in zip(plt_lines, body_trajectories):
        line.set_data(data[0:2, max(0, time_step-STEPS_TO_PLOT):time_step])
        line.set_3d_properties(data[2, max(0, time_step-STEPS_TO_PLOT):time_step])
        line.set(alpha=0.5)  # Make trajectories semi-transparent
    return plt_lines


def get_rotation_matrix(angle):
    """
    Constructs rotation matrix from 3d angle
    :param angle: 3d angle representing rotation around x, y, and z axes
    :return: Constructed 3d rotation matrix
    """
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
    """
    Construct random body orbiting at a given orbit

    1. Randomly choose the orbiting period
    2. Randomly choose body mass to be about 4*Pi^2*orbit_r^3 / (G*period^2)
    3. Randomly choose body velocity module to be about sqrt(G*CENTER_MASS/orbit_r),
       where CENTRAL_MASS is the mass of large body around which small bodies rotate
    4. Calculate body's initial position and velocity according to given initial orbit_angle

    :param orbit_r: Orbit radius on the body being constructed
    :param orbit_angle: Orbit angle represented as 3d rotation matrix
    :return: New body
    """

    # 1. Orbiting period
    period = np.random.uniform(BODY_PERIOD_LO, BODY_PERIOD_HI)

    # 2. Body mass
    m_exp = 4.0 * np.pi * np.pi * orbit_r**3 / (G * period * period)
    m_sigma = m_exp / 3.0
    m = np.random.normal(m_exp, m_sigma)

    # 3. Body velocity module
    v_exp = np.sqrt(G * CENTER_MASS / orbit_r)
    v_sigma = v_exp / 3.0
    v = np.random.normal(v_exp, v_sigma)

    # 4. Initial position and velocity
    velocity = orbit_angle @ np.asarray([0.0, v, 0.0])
    pos = orbit_angle @ np.asarray([orbit_r, 0.0, 0.0])

    return m, pos, velocity


def main():
    # Collection of random states generating nice trajectories. Uncomment those you want
    # np.random.seed(777)
    # np.random.seed(77777)
    # np.random.seed(777777)

    # Initializing bodies (mass, position, velocity)
    m = np.empty(N_BODIES)
    pos = np.empty((N_BODIES, 3))
    vel = np.empty((N_BODIES, 3))

    # Central body with mass much bigger than remaining bodies
    # As a result all other bodies will spin around this large object
    m[0] = CENTER_MASS  # Large mass
    pos[0] = np.zeros(3)  # Position at the center
    vel[0] = np.zeros(3)  # Zero velocity

    # Initialize remaining bodies
    for i in range(1, N_BODIES):
        orbit_radius = np.random.uniform(ORBIT_R_LO, ORBIT_R_HI)  # Random orbit
        orbit_angle = np.asarray([0.0, 0.0, np.random.uniform(0.0, 2.0*np.pi)])  # Random rotation around Z-axis
        rot_matrix = get_rotation_matrix(orbit_angle)  # Update rotation matrix accordingly
        bm, bp, bv = generate_body(orbit_radius, rot_matrix)  # Generate new body
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
    _ = animation.FuncAnimation(fig, update_lines, TIME_STEPS,
                                         fargs=(trajectories, axis_lines), interval=1, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
