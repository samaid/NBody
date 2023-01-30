import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time

from settings import *
from nbody_physics import *

def main():
    # Collection of random states generating nice trajectories. Uncomment those you want
    # np.random.seed(777)
    # np.random.seed(77777)
    np.random.seed(777777)

    t1 = time.time()
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

    # Generate trajectories
    trajectories = np.empty((N_BODIES, 3, TIME_STEPS))
    for time_step in range(TIME_STEPS):
        trajectories[..., time_step] = get_next_time_step_pos(bodies)

    t2 = time.time()
    print("Elapsed time for generating trajectories (seconds):", t2-t1)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

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
                                fargs=(trajectories, axis_lines), interval=10, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
