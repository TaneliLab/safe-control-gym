import numpy as np

import copy

import scipy.interpolate as interpol

import scipy.optimize as opt

import matplotlib.pyplot as plt

class LocalReplanner:

    def __init__(self, spline, sampleRate, current_gateID, current_gate_pos):
        
        # sampleRate: to get 
        self.spline = spline
        self.coeffs = spline.c
        self.knot = spline.t
        self.t = self.knot[-1]
        self.degree = 5
        self.current_gateID = current_gateID
        self.current_gate_pos = current_gate_pos
        self.sampleRate = sampleRate

    def gateID2controlPoint(self):
        if self.current_gateID >=0:
            coeffs_id = 2 + (self.current_gateID + 1)*self.sampleRate
        else:
            coeffs_id = 0
        return coeffs_id

    def hardGateSwitch(self):
        if self.current_gateID >=0:
            coeffs_id = self.gateID2controlPoint()
            self.coeffs[coeffs_id] = self.current_gate_pos[0:3]
            spline = interpol.BSpline(self.knot, self.coeffs, self.degree)
            return spline
        else:
            return False