#  MIT License
#  Copyright (c) 2021, Samuel Pell.
#  Author: Samuel Pell, sam.pell@canterbury.ac.nz
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""
    A library to store lots of small utilities useful for my communication simulation work.

    Author: Samuel Pell, sam.pell@canterbury.ac.nz
    Date:   15/12/21

    Change log:
        11/01/22 Add free space path loss calculator
        15/02/22 Plot the ECDF on the major axis of the figure passed in
        03/05/22 Add a version of the ECDF plotter which plots to an axis object passed in.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as consts


def deg_to_rad(deg):
    """Convert angular degrees to radians"""
    return deg * math.pi / 180


def rad_to_deg(rad):
    """Convert radians to angular degrees"""
    return rad * 180 / math.pi


def nat_to_db(nat):
    """Convert a natural number to decibels"""
    return 10.0 * np.log10(nat)


def db_to_nat(db):
    """Convert a number in decibels into a natural number"""
    return np.power(10, np.divide(db, 10))


def plot_ecdf(fig, data, label=""):
    fig.gca().plot(np.sort(data), np.linspace(0, 1, len(data)), label=label)


def plot_ecdf_axes(ax, data, label=""):
    ax.plot(np.sort(data), np.linspace(0, 1, len(data)), label=label)


def map_err(err):
    """
       Error on the prediction can only be between 0 - 180 due to the nature of circles. This function corrects the
       absolute error to absolute relative error - what we are interested in.
    """
    if err > 180:
        return (180 - err) % 180
    else:
        return err


def map_err2(est, real):
    """
       Error on the prediction can only be between 0 - 180 due to the nature of circles. This function corrects the
       absolute error to absolute relative error - what we are interested in.
    """
    err = np.abs(est - real)
    if err > 180:
        return 180 - (err % 180)
    else:
        return err


def freespace_path_loss(d, f):
    """
       Calculate the free-space path loss based on the distance between transmitter and receiver and the frequency of
       the transmission.

       Inputs:
           - d: Distance between transmitter and receiver in metres (m).
           - f: Frequency of the signal which is being attenuated in hertz (Hz).
    """
    return 20 * (np.log10(d) + np.log10(f) + np.log10(4 * consts.pi / consts.speed_of_light))
