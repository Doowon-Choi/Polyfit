# -*- coding: utf-8 -*-

import numpy as np
from .polyfit import load_example, PolynomRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

x, y = load_example()
x_plot = np.linspace(0, 90, 400)

DEG = 3
VERBOSE = False
datarange=[0, 1]

np_coeffs = np.polyfit(x, y, 5)
print("np: ", np_coeffs)
polyestimator = PolynomRegressor(deg=5)
vander = polyestimator.vander(x_plot)
pred_numpy = vander@np_coeffs[::-1]

polyestimator.fit(x, y) 
pred = polyestimator.predict(x_plot)
'''
#robust polynom fitting with L1 loss with crossvalidation to find best degree
degrees = range(1, 7)
polyestimator = PolynomRegressor(regularization='l1')#, monotonocity='positive')
poly_l1 = GridSearchCV(polyestimator, param_grid={'deg': degrees, 'lam': np.logspace(-4, 0, 20)}, \
                        scoring='neg_median_absolute_error', n_jobs = 3)
poly_l1.fit(X, y, groups=None, **{'loss':'l1'})#, , 'verbose':True
print("L1 regulaization Monotonic. Best degree: {}, best lambda: {}".format(poly_l1.best_params_['deg'], \
                                                poly_l1.best_params_['lam']))
pred_l1 = poly_l1.predict(x_plot)
#print("pred mon: ", pred_mon)
#robust polynom fitting with huber loss under shape and prediction constraints
degrees = range(1, 7)
polyestimator = PolynomRegressor(monotonocity='positive', \
                                 regularization='l1')##curvature='concave', 
poly_constrained = GridSearchCV(polyestimator, \
                        param_grid={'deg': degrees, 'lam': np.logspace(-4, 0, 20)},
                        scoring='neg_median_absolute_error', n_jobs = 3)
poly_constrained.fit(X, y, groups=None, **{'loss':'l1', 'yrange':datarange})
print("L1 regulaization Concave. Best degree: {}, best lambda: {}".format(poly_constrained.best_params_['deg'], \
                                                poly_constrained.best_params_['lam']))
#print("concave coeffs: ", poly_constrained.best_estimator_)
pred_concave = poly_constrained.predict(x_plot)
#polyestimator.fit(X, y, loss = 'huber', m = 0.05, yrange=datarange, verbose = VERBOSE)
#pred_concave=polyestimator.predict(x_plot)
#print("constrained coeffs: ", polyestimator.coeffs_)

#robust polynom fitting with huber loss under monotonicity and prediction constraints
#polyestimator = PolynomRegressor(deg=DEG, monotonocity='positive')
#polyestimator.fit(X, y, loss = 'huber', m = 0.05, yrange=datarange, verbose = VERBOSE)
#pred_mon=polyestimator.predict(x_plot)
'''
f, ax = plt.subplots(1, figsize = (10, 5))
ax.set_xlim(np.amin(x_plot), np.amax(x_plot))
#ax.set_ylim(-5, 5)

ax.scatter(x, y, c='k', s=8)

ax.plot(x_plot, pred_numpy, c='r', label='Deg: 5 /Numpy Polyfit')
ax.plot(x_plot, pred, c='b', label='Polyfit')

#ax.plot(x_plot, pred_l1, c='r', label='Crossvalidated L1 Loss')
#ax.plot(x_plot, pred_mon, c='r', label='Deg: 3 /Monotonic/Bounded Huber Loss')
#ax.plot(x_plot, pred_concave, c='b', label='Crossvalidated Monotonic/Bounded L1 Loss')

'''
axins = zoomed_inset_axes(ax, 2, loc='lower right', borderpad=1.5)
axins.set_xlim(55, 84) # apply the x-limits
axins.set_ylim(0.93, 1.028)

axins.yaxis.set_visible(False)
axins.xaxis.set_visible(False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0")

axins.scatter(X, y, c='k', s=8)
axins.plot(x_plot, pred_numpy, c='g')
#axins.plot(x_plot, pred_unconstrained, c='k')
axins.plot(x_plot, pred_l1, c='r')
axins.plot(x_plot, pred_concave, c='b')
'''
ax.legend(loc='upper left', frameon=False)
plt.subplots_adjust(top=0.99,
bottom=0.06,
left=0.05,
right=0.99)
plt.show()