import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import json
from solver import lqr_ref_tracking, process


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED


def solvelqrtracking(data):
    psi = float(data['psi'])
    x, y = float(data['x']), float(data['y'])
    speed = float(data['speed'])
    sa = float(data['steering_angle']) + 0.00001
    throttle = float(data['throttle'])
    lf = 2.67
    v = 20
    # Constant throttle 0.25
    A = np.matrix([[0, 0, 0], [0, 0, v], [0, 0, 0]])
    B = np.matrix([[1, 0], [0, 0], [1/(lf*sa), 0]])
    Q = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.matrix([[1.0, 0.0], [0.0, 1.0]]) * 0.1
    dt = 0.1
    t = 0.0
    initial_state = np.matrix([x, y, psi]).T
    initial_u = np.matrix([throttle, sa]).T
    ref_state = np.matrix(
        [data['ptsx'][0], data['ptsy'][0], 0]).T
    uref = np.matrix([0.5, 0.0]).T
    X = initial_state   # np.matrix([0, 0, 0]).T
    u1 = None
    while t <= 1.0:
        u = lqr_ref_tracking(X, ref_state, uref, A, B, Q, R, dt)
        if u1 is None:
            u1 = u
        X = process(A, B, X, u)
        t += dt
    # print ('input from solver', u1)
    return u1

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Tracking points
        # print ('telemetry', sid, data)
        speed = float(data["speed"])
        u1 = solvelqrtracking(data)
        try:
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            # throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            # steering_angle = np.random.choice(
            #    np.linspace(
            #        np.pi/2 - np.pi/18, np.pi/2 + np.pi/18, 10))
            # throttle = 0.25

            send_control(
                u1[1][0].item(), u1[0][0].item(), data['ptsx'], data['ptsy'])
            # send_control(0, 1)
        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)


def send_control(steering_angle, throttle, ptsx, ptsy):
    jdata = {
            'steering_angle': steering_angle,
            'throttle': throttle,
            'mpc_x': ptsx,
            'mpc_y': ptsy,
            'next_x': ptsx,
            'next_y': ptsy
        }
    sio.emit(
        "steer",
        data=jdata,
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPC with Linear Model')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
