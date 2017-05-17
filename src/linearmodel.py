
import argparse
from os import listdir
from os.path import isfile, join, splitext
from ntpath import basename
import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
from path import *
import csv
from multiprocessing.pool import ThreadPool
from functools import partial
from sklearn import datasets, linear_model
import pickle

def linearRegressionModelTrain(X_train, X_test, vly_train, vly_test, agy_train, agy_test, **options):
    rseg = options['rseg']
    cseg = options['cseg']
    objmask = options['objmask']
    parampath = SCRATCH_PATH
    if 'parampath' in options:
        parampath = options['parampath']
    X_train = np.reshape(X_train, (X_train.shape[0],-1))
    X_test = np.reshape(X_test, (X_test.shape[0],-1))
    # Create linear regression object
    regr_speed = linear_model.LinearRegression(fit_intercept=True)
    # Train the model using the training sets
    regr_speed.fit(X_train, vly_train)
    vlmse = np.mean((regr_speed.predict(X_test) - vly_test) ** 2)
    vlvar = regr_speed.score(X_test, vly_test)

    agy_train = np.rad2deg(agy_train)
    agy_test = np.rad2deg(agy_test)
    # Create linear regression object
    regr_angle = linear_model.LinearRegression(fit_intercept=True)
    # Train the model using the training sets
    regr_angle.fit(X_train, agy_train)
    agmse = np.mean((regr_angle.predict(X_test) - agy_test) ** 2)
    agvar = regr_speed.score(X_test, agy_test)

    # write coefficients into a file
    params = {}
    params['rseg'] = rseg
    params['cseg'] = cseg
    params['speed_coef'] = regr_speed.coef_
    params['angle_coef'] = regr_angle.coef_
    om = 'om_1' if objmask else 'om_0'
    pickle.dump(params , open('{}/linear_reg_params_{}.pickle'.format(parampath, om), "wb"))
    # with open('{0}/parameters.txt'.format(parampath), 'w') as paramfile:
        # paramfile.write(','.join(map(str, [rseg, cseg])) + '\n')
        # paramfile.write(','.join(map(str, regr_speed.coef_)) + '\n')
        # paramfile.write(','.join(map(str, regr_angle.coef_)) + '\n')

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    return (vlmse, vlvar, agmse, agvar)

def linearRegressionModelTest(X_test, **options):
    parampath = SCRATCH_PATH
    if 'parampath' in options:
        parampath = options['parampath']

    X_test = np.reshape([X_test], (1,-1))
    # load parameters from file
    params = pickle.load(open('{}/linear_reg_params.pickle'.format(parampath), "rb" ))
    rseg = params['rseg']
    cseg = params['cseg']
    options['rseg'] = rseg
    options['cseg'] = cseg
    coef_speed = params['speed_coef']
    coef_angle = params['angle_coef']
    # with open('{0}/parameters.txt'.format(parampath), 'r') as paramfile:
        # rseg, cseg = paramfile.readline().split(',')
        # rseg = int(rseg)
        # cseg = int(cseg)
        # coef_speed = paramfile.readline().split(',')
        # coef_speed = np.array(map(float, coef_speed))
        # coef_angle = paramfile.readline().split(',')
        # coef_angle = np.array(map(float, coef_angle))
    regr_speed = linear_model.LinearRegression()
    regr_speed.coef_ = coef_speed
    regr_speed.intercept_ = True
    speed = regr_speed.predict(X_test)[0]
    regr_angle = linear_model.LinearRegression()
    regr_angle.coef_ = coef_angle
    regr_angle.intercept_ = True
    angle = regr_angle.predict(X_test)[0]

    return speed, angle

