import numpy as np

from settings import *


class Camera:
    def __init__(self, camera_pos, camera_angle, display_pos):
        self.camera_pos = camera_pos
        self.camera_angle = camera_angle
        self.display_pos = display_pos
        self.rot_x = np.empty((3, 3))
        self.rot_y = np.empty((3, 3))
        self.rot_z = np.empty((3, 3))
        self.update_rotation_matrices()

    def update_rotation_matrices(self):
        theta = self.camera_angle
        cx = np.cos(theta[0])
        sx = np.sin(theta[0])
        cy = np.cos(theta[1])
        sy = np.sin(theta[1])
        cz = np.cos(theta[2])
        sz = np.sin(theta[2])

        self.rot_x = np.array([[1.0, 0.0, 0.0],
                               [0.0, cx, sx],
                               [0.0, -sx, cx]])
        self.rot_y = np.array([[cy, 0.0, -sy],
                               [0.0, 1.0, 0.0],
                               [sy, 0.0, cy]])
        self.rot_z = np.array([[cz, sz, 0.0],
                               [-sz, cz, 0.0],
                               [0.0, 0.0, 1.0]])

    def to_screen(self, pos):
        dd = pos - self.camera_pos
        rot_xyz = (self.rot_y * self.rot_z) * self.rot_x
        d = dd @ rot_xyz.T

        bx = self.display_pos[2] * d[:, 0] / d[:, 2] + self.display_pos[0]
        by = self.display_pos[2] * d[:, 1] / d[:, 2] + self.display_pos[1]

        x = np.rint(bx).astype(int)
        y = np.rint(by).astype(int)
        return x, y


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
        softening_sq = 0.01
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
        inv_r3 = (r2 + softening_sq)**(-1.5)

        ax = G * (dx * inv_r3) @ mass
        ay = G * (dy * inv_r3) @ mass
        az = G * (dz * inv_r3) @ mass

        self.acceleration = np.stack((ax, ay, az), axis=1)
        self.velocity += self.acceleration * dt * 0.5

    def draw(self, surface, camera):
        bx, by = camera.to_screen(self.pos)

        d = self.pos - camera.camera_pos
        dx2 = d[:, 0] * d[:, 0]
        dy2 = d[:, 1] * d[:, 1]
        dz2 = d[:, 2] * d[:, 2]
        r = np.sqrt(dx2 + dy2 + dz2)
        rr = np.rint(80.0/r).astype(int)

        for i in range(bx.shape[0]):
            pg.draw.circle(surface, "white", (bx[i], by[i]), rr[i])


def main():
    ds, clk = initialize()

    m = np.asarray([1000.0, 1.0, 1.0, 1.0])

    pos = np.asarray([[0.0, 0.0, 0.0],
                     [-10.0, 0.0, 0.0],
                     [-20.0, 0.0, 0.0],
                     [-30.0, 0.0, 0.0]])

    vel = np.asarray([[0.0, 0.0, 0.0],
                      [0.0, 10.0, 0.0],
                      [0.0, 6.0, 0.0],
                      [2.0, 2.0, 2.0]])

    bodies = Bodies(m, pos, vel)

    camera = Camera(np.array([0.0, 0.0, 10.0]),
                    np.array([0.0, 0.8, 0.0]),
                    np.array([600.0, 400.0, 100.0]))

    transparent = pg.Surface(DISPLAY_RES)
    transparent.set_alpha(80)

    do_game = True
    while do_game:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                do_game = False

        # Draw objects
        transparent.fill("black")
        bodies.draw(transparent, camera)

        ds.blit(transparent, (0, 0))

        # Perform updates
        bodies.update()

        # Prepare for next frame
        pg.display.flip()
        clk.tick(FPS)
    pg.quit()


if __name__ == "__main__":
    main()
