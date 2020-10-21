#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:25:16 2020

@author: tyrion
"""
import numpy as np
import cvxpy as cv
from sklearn.base import BaseEstimator
from os.path import dirname
import time

PATH = dirname(__file__)

def load_example():
    
    npzfile = np.load(PATH + '/Example_Data.npz')
    X = npzfile['X']
    y = npzfile['y']
    
    return X, y

class PolynomRegressor(BaseEstimator):
    
    def __init__(self, deg=None, monotonocity = None, curvature = None, \
                 positive_coeffs = False, negative_coeffs = False, \
                     regularization = None, lam = 0):
        
        self.deg = deg
        self.monotonocity = monotonocity
        self.curvature = curvature
        self.coeffs_ = None
        self.positive_coeffs = positive_coeffs
        self.negative_coeffs = negative_coeffs
        self.regularization = regularization
        self.lam = lam
    
    def column_norms(self, V):
        
        norms = np.sqrt(np.square(V).sum(0))
        
        norms[norms == 0] = 1
        
        return norms
    
    def vander(self, x):
        
        x = x.astype(np.float64)
        return np.fliplr(np.vander(x, N = self.deg +1))

    def vander_grad(self, x):
        
        vander = self.vander(x)
        
        red_vander = vander[:, :-1]
        
        factors = np.arange(1, self.deg+1)
        
        grad_matrix = np.zeros(shape=vander.shape)
        inner_matrix = red_vander * factors[None, :]
        grad_matrix[:, 1:] = inner_matrix
        
        return grad_matrix

    def vander_hesse(self, x):
        
        grad = self.vander_grad(x)
        
        red_grad = grad[:, :-1]
        
        factors = np.arange(1, self.deg+1)
        
        hesse_matrix = np.zeros(shape=grad.shape)
        inner = red_grad * factors[None, :]
        hesse_matrix[:, 2:] = inner[:, 1:]
        
        return hesse_matrix
        
    def predict(self, x):
        
        if self.coeffs_ is not None:
            
            vander = self.vander(x)
    
            return np.dot(vander, self.coeffs_)
        
        else:
            
            return np.nan

    def check_monotonicity(self, coeffs, constraint_range):
        '''
        Returns roots of first derivative


        Parameters
        ----------
        coeffs : ndarray

        Returns
        -------
        roots : ndarray.

        '''        
        reversed_coeffs = coeffs[::-1]
        
        #calculate coefficients of 1. derivative and find roots of the polynomial    
        derivative_coeffs = np.polyder(reversed_coeffs)
        
        roots = np.roots(derivative_coeffs)
        
        #check if roots are complex
        complex_roots = np.all(np.iscomplex(roots))

        grad_real_roots = roots[np.isreal(roots)]

        if complex_roots:
        
            monotonic = True
            roots_within_constraints = None
        else:
            
            #check if real roots lie within the range where monotonicity\
                #should be enforced
            #print(real_roots)
            roots_within_constraints_indices = (grad_real_roots > constraint_range[0]) & (grad_real_roots < constraint_range[1])
            roots_within_constraints = grad_real_roots[roots_within_constraints_indices]
            
            if roots_within_constraints.size == 0:
                
                monotonic = True
                
            else:
                
                monotonic = False            
        return roots_within_constraints, monotonic
    
    def check_curvature(self, coeffs, constraint_range):
        '''
        Returns roots of second derivative


        Parameters
        ----------
        coeffs : ndarray

        Returns
        -------
        roots : ndarray.

        '''        
        reversed_coeffs = coeffs[::-1]
        
        #calculate coefficients of 1. derivative and find roots of the polynomial    
        second_derivative_coeffs = np.polyder(reversed_coeffs, m = 2)
        
        roots = np.roots(second_derivative_coeffs)
        
        #check if roots are complex
        complex_roots = np.all(np.iscomplex(roots))
        
        hesse_real_roots = roots[np.isreal(roots)]
        #print("hesse real roots: ", hesse_real_roots)
        if complex_roots:
            
            strict_curvature = True
            hesse_roots_within_constraints = None
        else:
            
            #check if real roots lie within the range where constraints\
                #should be enforced
            hesse_roots_within_constraints_indices = (hesse_real_roots > constraint_range[0]) & (hesse_real_roots < constraint_range[1])
            hesse_roots_within_constraints = hesse_real_roots[hesse_roots_within_constraints_indices]
            
            if hesse_roots_within_constraints.size == 0:
                
                strict_curvature = True
                
            else:
                
                strict_curvature = False
        #print("hesse roots within boundaries: ", hesse_real_roots[hesse_roots_within_constraints_indices])
        return hesse_roots_within_constraints, strict_curvature
        

    def build_objective(self, x, y, loss = 'l2', m = 1):
        
        vander = self.vander(x)
        #print("vander shape: ", vander.shape)
        column_norms_vander = self.column_norms(vander)
        vander = vander/column_norms_vander
        
        #set up variable for coefficients to be estimated
        
        if self.positive_coeffs:
            
            coeffs = cv.Variable(self.deg +1, pos = True)
        
        elif self.negative_coeffs:
            
            coeffs = cv.Variable(self.deg +1, neg = True)
            
        else:
            
            coeffs = cv.Variable(self.deg +1)
        
        #print(coeffs)
        #calculate residuals
        
        residuals = vander @ coeffs -y
        
        #define loss function
        
        if self.regularization == 'l1':
            
            regularization_term = cv.norm1(coeffs)
            
        elif self.regularization == 'l2':
            
            regularization_term = cv.pnorm(coeffs, 2, axis = 0)**2
        
        else:
            
            regularization_term = 0
            
        if loss == 'l2':
            
            data_term = cv.sum_squares(residuals)
                                    
        elif loss == 'l1':
            
            data_term = cv.norm1(residuals)
        
        elif loss == 'huber':
            
            data_term = cv.sum(cv.huber(residuals, m))
        
        objective = cv.Minimize(data_term + self.lam * regularization_term)

        return objective, coeffs        
    
    def build_monotonic_constraints(self, coeffs, roots, column_norms):
        
        constraints = []
        
        #vandergrad = self.vander_grad(roots)
        for root in roots:
            #print("constraint for root: ", root)
            vandergrad = self.vander_grad(np.array([root]))
            vandergrad = vandergrad.ravel()/column_norms
            derivative = cv.sum(cv.multiply(vandergrad, coeffs))
            
            if self.monotonocity == 'positive':
                constraints.append(derivative >= 1e-3)
            
            elif self.monotonocity == 'negative':
                constraints.append(derivative <= -1e-3)
                
        return constraints
    
    def build_curvature_constraints(self, coeffs, roots, column_norms):
        
        constraints = []
        if roots is None:
            
            return constraints
        
        #vandergrad = self.vander_grad(roots)
        for root in roots:
            #print("constraint for root: ", root)
            vanderhesse = self.vander_hesse(np.array([root]))
            vanderhesse = vanderhesse.ravel()/column_norms
            second_derivative = cv.sum(cv.multiply(vanderhesse, coeffs))
            
            if self.curvature == 'convex':
                constraints.append(second_derivative >= 1e-2)
            
            elif self.curvature == 'concave':
                constraints.append(second_derivative <= -1e-2)
                
        return constraints
    
    def build_range_constraint(self, coeffs, constraint_range, prediction_range, \
                               column_norms):
        
        constraints = []
        
        for x in constraint_range:
            
            vanderx = self.vander(np.array([x]))
            vanderx = vanderx.ravel()/column_norms
            
            prediction = cv.sum(cv.multiply(vanderx, coeffs))
            
            constraints.append(prediction >= prediction_range[0])
            constraints.append(prediction <= prediction_range[1])
            
        return constraints

    def cvx_solve(self, problem, coeffs, loss, verbose):
        print("loss: ", loss)
        try:
            
            if loss == 'l1' or self.regularization is not None:
            #l1 loss solved by ECOS. Lower its tolerances for convergence    
                problem.solve(abstol=1e-9, reltol=1e-9, max_iters=1000000, \
                              feastol=1e-9, abstol_inacc = 1e-7, \
                                  reltol_inacc=1e-7, verbose = verbose)            
                
            else:
                    
                #l2 and huber losses solved by OSQP. Lower its tolerances for convergence
                problem.solve(eps_abs=1e-10, eps_rel=1e-10, max_iter=10000000, \
                              eps_prim_inf = 1e-10, eps_dual_inf = 1e-10, verbose = verbose) 
                    
        #in case OSQP or ECOS fail, use SCS
        except cv.SolverError:
            
            try:
            
                problem.solve(solver=cv.SCS, max_iters=100000, eps=1e-4, verbose = verbose)
            
            except cv.SolverError:
                    
                print("cvxpy optimization failed!")
        
        #if optimal solution found, set parameters
        
        if problem.status == 'optimal':
            
            return problem, coeffs.value
        
        #if not try SCS optimization
        else:
            
            try:
                
                problem.solve(solver=cv.SCS, max_iters=100000, eps=1e-6, verbose = verbose)
            
            except cv.SolverError:
                
                pass
        
        if problem.status == 'optimal':

            return problem, coeffs.value
            
            #coefficients = coeffs.value/column_norms_vander

            #self.coeffs_ = coefficients        
        
    def fit(self, x, y, loss = 'l2', m = 1, constraint_range = None, yrange = None, verbose = False):
        
        if constraint_range is None:
            
            constraint_range = [np.amin(x), np.amax(x)]
        

        vander = self.vander(x)
        column_norms = self.column_norms(vander)
        
        objective, coeffs = self.build_objective(x, y, loss, m)
        
        constraints = []
        if yrange is not None:
            
            y_constraints = self.build_range_constraint(coeffs, constraint_range, yrange, column_norms)
            constraints = constraints + y_constraints
            
        problem = cv.Problem(objective, constraints)
        problem.solve()
        unscaled_coeffs = coeffs.value
        rescaled_coeffs = unscaled_coeffs/column_norms
        #problem, coeffs = self.cvx_solve(problem, coeffs, loss, verbose)
        

        if self.curvature is None:
            
            curvature = True
            
        if self.monotonocity is not None:
            
            gradient_roots, monotonic = self.check_monotonicity(rescaled_coeffs, constraint_range)
            hesse_roots, curva = self.check_curvature(rescaled_coeffs, constraint_range)
            print("initial hesse root: ", hesse_roots)
            while monotonic == False or curvature == False:
                
                #print("monotonic: ", monotonic)
                #print("derivative roots: ", gradient_roots)
                #print("current coefs: ", coeffs.value)
                #print("curvature: ", curvature)
                #print("hesse rots: ", hesse_roots)
                #time.sleep(0.25)
                monotonic_constraints = self.build_monotonic_constraints(coeffs, gradient_roots, column_norms)
                curvature_constraints = self.build_curvature_constraints(coeffs, hesse_roots, column_norms)
                constraints = y_constraints + monotonic_constraints + curvature_constraints
                #print(constraints)
                problem = cv.Problem(objective, constraints = constraints)
                #print(problem)
                problem.solve(verbose = True)#eps_abs=1e-10, eps_rel=1e-10, max_iter=10000000, \
                              #eps_prim_inf = 1e-10, eps_dual_inf = 1e-10, verbose = verbose)
                unscaled_coeffs = coeffs.value
                rescaled_coeffs = unscaled_coeffs/column_norms
                #problem, coeffs = self.cvx_solve(problem, coeffs, loss, verbose)
                
                gradient_roots, monotonic = self.check_monotonicity(rescaled_coeffs, constraint_range)
                #hesse_roots, curvature = self.check_curvature(rescaled_coeffs, constraint_range)
    
        #rescaled_coeffs = coeffs/column_norms
        print("cvx solution: ", rescaled_coeffs)
        #print("hesse root: ", hesse_roots)
        self.coeffs_ = rescaled_coeffs
        
        return self

npzfile = np.load('/home/tyrion/Development/Polyfit/polyfit/Example_Data.npz')
x = npzfile['X']
y = npzfile['y']
x_plot = np.linspace(0, 100, 500)    
DEG = 7
VERBOSE = False
datarange=[0, 1]

np_coeffs = np.polyfit(x, y, DEG)
#der_coeffs = np.
print("np: ", np_coeffs)
polyestimator = PolynomRegressor(deg=DEG, monotonocity='positive')#, curvature='concave')
vander = polyestimator.vander(x_plot)
pred_numpy = vander@np_coeffs[::-1]

polyestimator.fit(x, y, loss = 'l1', m = 0.05, yrange = [0,1], verbose = False) #, 
pred = polyestimator.predict(x_plot)

import matplotlib.pyplot as plt

f, ax = plt.subplots(1, figsize = (10, 5))
ax.set_xlim(np.amin(x_plot), np.amax(x_plot))
ax.set_ylim(0, 1.1)

ax.scatter(x, y, c='k', s=8)

ax.plot(x_plot, pred_numpy, c='r', label='Deg: 5 /Numpy Polyfit')
ax.plot(x_plot, pred, c='b', label='Polyfit')
ax.legend(loc='upper left', frameon=False)
plt.subplots_adjust(top=0.99,
bottom=0.06,
left=0.05,
right=0.99)
plt.show()