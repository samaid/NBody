import os
import pygame as pg

DISPLAY_RES = DISPLAY_W, DISPLAY_H = 1200, 800
WORLD_TO_SCREEN_COORDINATES = 1.0

FPS = 60

G = 1.0
CENTER_MASS = 100000.0
BODY_MASS_LO = 0.5
BODY_MASS_HI = 5.0
ORBIT_R_LO = 10.0
ORBIT_R_HI = 100.0
DT = 0.1
TIME_STEPS = 500
N_BODIES = 40


def set_display():
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    pg.init()
    surface = pg.display.set_mode(DISPLAY_RES, pg.SCALED)
    clock = pg.time.Clock()

    return surface, clock


def initialize():
    surface, clock = set_display()
    return surface, clock


def to_screen(world):
    return int(world*WORLD_TO_SCREEN_COORDINATES)
