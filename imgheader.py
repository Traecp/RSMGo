#!/usr/bin/env python
# -*- coding: utf-8 -*-

def get_counter(imgheader):
    ctname = imgheader["counter_mne"].split()
    ctval  = imgheader["counter_pos"].split()
    counter = {}
    for i in range(len(ctname)):
        counter[ctname[i]] = float(ctval[i])
    return counter
    
def get_motor(imgheader):
    ctname = imgheader["motor_mne"].split()
    ctval  = imgheader["motor_pos"].split()
    motor = {}
    for i in range(len(ctname)):
        motor[ctname[i]] = float(ctval[i])
    return motor