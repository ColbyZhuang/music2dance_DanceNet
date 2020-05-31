# -*- coding: utf-8 -*-
import numpy as np
import math


def degree2radian(theta):
    return theta * np.pi / 180.0


def radian2degree(theta):
    return theta / np.pi * 180.0


def rotation_x(theta_x):
    theta_x = degree2radian(theta_x)
    rx = np.mat([[1.0, 0.0, 0.0, 0.0],
                 [0.0, np.cos(theta_x), -np.sin(theta_x), 0.0],
                 [0.0, np.sin(theta_x), np.cos(theta_x), 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

    return rx


def rotation_y(theta_y):
    theta_y = degree2radian(theta_y)
    ry = np.mat([[np.cos(theta_y), 0.0, np.sin(theta_y), 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [-np.sin(theta_y), 0.0, np.cos(theta_y), 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

    return ry


def rotation_z(theta_z):
    theta_z = degree2radian(theta_z)
    rz = np.mat([[np.cos(theta_z), -np.sin(theta_z), 0.0, 0.0],
                 [np.sin(theta_z), np.cos(theta_z), 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

    return rz


def rotation_from_mat(rx, ry, rz):
    return rz * ry * rx


def rotation_from_angle(theta_x, theta_y, theta_z):
    rx = rotation_x(theta_x)
    ry = rotation_y(theta_y)
    rz = rotation_z(theta_z)

    return rz * ry * rx


def translation(tx, ty, tz):
    t = np.mat([[1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, ty],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0]])

    return t


def rotation2angle(r):
    """
    :param r:  rotation matrix
    :return: euler angle in radian
    """

    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    rotate_angle = np.array([0.0, 0.0, 0.0])
    singular = sy < 1e-6
    if not singular:
        rotate_angle[0] = math.atan2(r[2, 1], r[2, 2]) / np.pi * 180.0
        rotate_angle[1] = math.atan2(-r[2, 0], sy) / np.pi * 180.0
        rotate_angle[2] = math.atan2(r[1, 0], r[0, 0]) / np.pi * 180.0
    else:
        rotate_angle[0] = math.atan2(-r[1, 2], r[1, 1]) / np.pi * 180.0
        rotate_angle[1] = math.atan2(-r[2, 0], sy) / np.pi * 180.0
        rotate_angle[2] = 0

    return rotate_angle
