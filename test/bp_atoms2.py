
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


Points_0 = [
        
COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.013,-28.524,1.803,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.122,-28.368,-0.261,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.249,-27.162,-0.47,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.381,-28.872,0.725,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.486,-28.049,1.52,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.166,-27.966,0.774,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.708,-28.966,0.22,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.27,-28.691,2.893,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.486,-28.589,3.843,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.294,-29.43,5.089,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.746,-30.797,4.877,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.567,-26.772,0.729,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.344,-26.594,-0.056,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.168,-26.041,0.729,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.029,-26.465,0.504,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.63,-25.684,-1.239,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.778,-26.19,-2.101,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.135,-25.225,-3.195,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.612,-24.118,-2.933,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.877,-25.622,-4.441,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.4,-25.106,1.621,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.323,-24.559,2.375,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.738,-24.146,3.762,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.792,-23.637,3.904,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.793,-23.349,1.614,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.953,-22.445,2.444,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.175,-21.466,1.634,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-78.678,-20.449,2.633,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-77.275,-20.038,2.567,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.909,-24.366,4.76,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.188,-23.971,6.137,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.978,-22.476,6.295,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.918,-21.951,5.951,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.281,-24.736,7.106,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.449,-24.194,8.536,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.603,-26.203,7.058,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.98,-21.788,6.847,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.914,-20.345,7.061,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.196,-19.968,8.503,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.164,-18.786,8.85,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.842,-19.577,6.101,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.313,-19.896,6.399,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.533,-19.974,4.677,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.275,-19.202,5.48,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.456,-20.947,9.349,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.636,-20.728,10.757,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.582,-22.079,11.419,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.804,-23.112,10.776,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.277,-22.069,12.706,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.152,-23.363,13.346,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.288,-23.188,14.845,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.076,-22.095,15.38,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.84,-24.069,12.962,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.692,-24.273,15.496,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.805,-24.356,16.934,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.18,-28.043,14.366,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.401,-26.915,13.734,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.617,-26.23,14.393,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.625,-26.734,12.437,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.982,-25.632,11.746,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.933,-24.981,10.742,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.069,-25.412,10.537,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.604,-26.018,11.191,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.567,-27.204,10.311,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.594,-28.541,11.058,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.195,-28.628,12.247,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.059,-29.51,10.429,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.489,-23.882,10.158,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.249,-23.183,9.128,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.459,-23.315,7.848,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.234,-23.14,7.865,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.408,-21.686,9.481,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.293,-20.937,8.45,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.9,-21.515,10.914,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.158,-23.545,6.738,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.541,-23.712,5.429,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.165,-22.762,4.422,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.364,-22.471,4.484,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.81,-25.127,4.877,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.32,-26.26,5.772,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.09,-26.857,5.547,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.101,-26.738,6.825,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.637,-27.89,6.343,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.66,-27.78,7.617,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.419,-28.35,7.367,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.95,-29.379,8.142,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.35,-22.305,3.465,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.887,-21.869,2.189,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.186,-23.12,1.38,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.384,-24.065,1.359,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.872,-21.018,1.412,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.438,-20.328,0.156,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.37,-21.195,-1.098,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.01,-20.542,-2.312,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.851,-21.457,-3.488,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.325,-23.115,0.712,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.7,-24.209,-0.154,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.63,-23.801,-1.272,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.851,-22.61,-1.548,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.201,-24.808,-1.936,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.175,-24.588,-2.998,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.445,-25.365,-2.674,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.389,-26.436,-2.056,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.608,-25.045,-4.335,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.294,-24.379,-4.718,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.505,-22.615,-5.04,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.877,-22.651,-6.179,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.583,-24.777,-3.021,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.908,-25.357,-2.818,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.804,-24.444,-1.956,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.162,-20.983,-5.585,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.199,-21.093,-4.405,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.219,-22.099,-3.686,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.563,-20.488,-5.171,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.047,-21.189,-3.932,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.584,-18.991,-4.963,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.292,-20.132,-4.203,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.443,-20.183,-3.009,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.282,-20.021,-1.758,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.271,-19.278,-1.741,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.498,-18.985,-3.196,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.226,-18.087,-4.124,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.919,-19.005,-5.08,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.865,-20.726,-0.696,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.559,-20.695,0.584,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.515,-20.829,1.675,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.372,-21.236,1.439,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.585,-21.85,0.747,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.826,-21.647,-0.104,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.933,-23.183,0.424,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.905,-20.446,2.884,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.142,-20.793,4.069,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.831,-21.939,4.777,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.064,-21.994,4.841,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.024,-19.604,5.041,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.016,-22.812,5.364,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.459,-24.079,5.911,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.857,-24.261,7.313,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.636,-24.358,7.462,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.921,-25.14,4.939,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.583,-25.022,3.574,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.109,-26.406,5.421,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.729,-25.564,2.427,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.708,-24.33,8.341,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.235,-24.513,9.714,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.641,-25.905,10.175,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.807,-26.261,10.038,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.809,-23.476,10.68,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.295,-23.635,12.114,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.055,-22.676,13.11,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.725,-21.227,12.871,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.303,-20.274,13.914,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.693,-26.688,10.698,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.994,-28.057,11.121,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.982,-28.203,12.64,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.413,-27.388,13.385,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.006,-29.061,10.536,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.711,-28.798,11.086,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.93,-28.934,9.033,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.048,-25.989,13.69,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.748,-26.84,12.861,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.131,-19.132,14.711,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.604,-18.959,14.648,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.887,-19.966,14.746,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.128,-17.796,14.452,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.588,-16.891,11.933,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.097,-17.662,11.255,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.382,-17.366,9.502,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.204,-9.794,8.437,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.828,-10.999,7.596,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.417,-11.233,6.541,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.843,-11.78,8.066,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.395,-12.936,7.272,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.571,-13.845,6.948,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.364,-14.187,7.828,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.353,-13.705,8.087,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.098,-12.896,8.195,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.02,-15.021,7.366,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.4,-12.745,6.872,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.65,-14.282,5.684,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.774,-15.082,5.194,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.475,-16.566,5.352,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.397,-17.032,4.979,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.968,-14.806,3.711,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-100.028,-15.69,3.071,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-99.273,-14.43,1.046,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.19,-14.553,0.282,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.795,-15.762,-0.116,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.505,-13.468,-0.101,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-99.453,-17.305,5.886,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-99.365,-18.755,6.055,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-99.813,-19.417,4.762,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-100.31,-19.16,7.183,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-99.035,-21.153,8.074,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.993,-20.313,4.233,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-99.397,-21.141,3.101,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.204,-21.505,2.208,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-97.561,-20.283,1.55,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.42,-19.685,0.435,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.468,-18.44,0.306,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.36,-24.402,6.575,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.084,-28.353,9.169,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.065,-27.2,9.089,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.658,-28.686,7.749,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.531,-29.675,7.665,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.816,-29.938,6.04,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.532,-25.987,9.185,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.33,-24.794,8.88,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.685,-24.137,7.677,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.46,-23.95,7.662,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.342,-23.795,10.054,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.939,-24.421,11.31,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.178,-22.546,9.704,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.619,-23.624,12.56,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.5,-23.772,6.688,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.032,-23.248,5.414,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.599,-21.847,5.257,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.811,-21.643,5.421,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.486,-24.159,4.259,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.867,-25.536,4.481,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.022,-23.607,2.901,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.817,-26.663,4.248,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.732,-20.885,4.947,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.14,-19.495,4.785,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.547,-18.918,3.519,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.657,-19.498,2.882,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.714,-18.572,5.952,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.284,-18.355,5.902,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.117,-19.154,7.281,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.993,-17.714,3.188,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.272,-17.004,2.137,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.798,-16.857,2.509,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.424,-16.866,3.683,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.893,-15.626,1.892,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.772,-14.688,3.077,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.45,-13.339,2.827,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.294,-12.759,1.736,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.181,-12.882,3.724,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.949,-16.735,1.505,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.508,-16.608,1.693,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.116,-15.149,1.563,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.345,-14.538,0.517,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.762,-17.429,0.648,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.311,-17.247,0.781,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.652,-17.653,1.946,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.569,-16.612,-0.208,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.308,-17.448,2.097,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.211,-16.424,-0.077,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.584,-16.851,1.095,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.234,-16.688,1.28,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.557,-14.581,2.629,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.132,-13.184,2.626,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.651,-13.158,2.22,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.791,-13.615,2.971,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.372,-12.577,4.011,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.861,-12.695,4.402,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.92,-11.605,3.496,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.451,-10.024,4.257,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.355,-12.589,1.048,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.06,-12.811,0.426,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.913,-12.102,1.154,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.764,-12.498,0.955,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.141,-12.317,-1.037,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.162,-13.05,-1.935,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.528,-12.372,-1.973,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.891,-11.563,-1.1,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.261,-12.643,-2.943,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.189,-11.073,1.952,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.148,-10.309,2.61,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.0,-10.576,4.092,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.129,-9.963,4.734,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.329,-8.813,2.352,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.024,-8.458,0.87,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.926,-8.737,0.362,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.99,-7.874,0.197,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.779,-11.51,4.627,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.517,-11.999,5.976,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.916,-11.004,7.034,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.774,-10.164,6.837,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.268,-11.136,8.189,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.651,-10.359,9.349,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.168,-8.931,9.173,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.031,-8.683,8.745,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.033,-10.943,10.63,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.043,-7.99,9.567,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.777,-6.575,9.369,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.538,-6.099,10.128,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.8,-5.244,9.62,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.975,-5.782,9.89,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.866,-4.278,9.935,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.686,-3.731,8.516,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.123,-3.698,10.508,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.289,-6.591,11.342,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.115,-6.095,12.064,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.811,-6.431,11.345,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.952,-5.562,11.144,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.083,-6.614,13.515,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.989,-8.126,13.634,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.439,-8.875,12.763,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.413,-8.553,14.669,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.668,-7.673,10.912,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.444,-8.062,10.216,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.374,-7.402,8.844,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-78.282,-7.036,8.383,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.441,-9.585,10.072,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-78.142,-10.174,9.612,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-78.146,-11.68,9.887,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-76.855,-12.364,9.387,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-76.784,-12.283,7.9,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.507,-7.295,8.174,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.581,-6.628,6.882,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.048,-5.203,6.959,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.143,-4.822,6.182,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.06,-6.635,6.461,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.352,-5.892,5.174,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-81.952,-6.386,3.921,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.025,-4.688,5.225,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.27,-5.645,2.751,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.366,-3.992,4.06,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.998,-4.497,2.826,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.525,-4.43,7.934,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-75.986,-6.79,5.872,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-75.59,-7.771,7.009,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-75.916,-9.195,6.803,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-76.482,-9.544,5.753,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-75.58,-9.962,7.712,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-78.155,-5.988,1.801,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.457,-6.782,15.482,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.403,-5.47,14.281,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.028,-6.914,13.96,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.272,-7.378,12.834,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.521,-7.673,14.935,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.192,-9.079,14.735,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.421,-9.922,14.447,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.26,-11.016,13.882,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.436,-9.638,15.958,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-85.252,-9.717,17.214,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-84.403,-10.252,18.383,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.628,-9.44,14.773,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.869,-10.156,14.498,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.68,-9.588,13.323,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.887,-9.869,13.202,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.708,-10.24,15.764,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.085,-11.176,16.78,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.52,-12.228,16.397,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.119,-10.798,18.074,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.044,-8.798,12.474,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.618,-8.262,11.252,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.845,-8.854,10.095,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.606,-8.959,10.15,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.487,-6.724,11.235,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.168,-6.06,12.441,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.09,-6.112,9.939,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.595,-6.313,12.552,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.578,-9.234,9.031,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.949,-9.856,7.89,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.139,-8.922,6.709,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.114,-8.161,6.664,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.527,-11.246,7.591,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.227,-12.342,8.618,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.135,-13.56,8.377,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.735,-12.757,8.474,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.194,-8.969,5.756,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.186,-8.0,4.655,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.199,-8.761,3.336,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.389,-9.685,3.158,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.942,-7.108,4.733,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.975,-6.058,3.63,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.926,-6.399,6.097,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.071,-8.336,2.389,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.088,-9.015,1.088,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.395,-8.166,0.018,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.852,-7.081,0.285,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.508,-9.49,0.695,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.414,-8.355,0.267,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.001,-7.215,0.045,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.708,-8.657,0.156,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.409,-8.682,-1.213,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.697,-8.02,-2.33,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.393,-6.762,-2.848,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-87.55,-9.035,-3.466,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.828,-9.429,-3.955,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.583,-6.39,-2.322,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.038,-4.528,-0.455,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-88.738,-3.754,0.736,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.92,-3.7,1.708,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.913,-4.59,1.551,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.058,-4.645,2.457,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.632,-5.355,3.725,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.013,-6.439,3.685,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.226,-5.386,1.813,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.92,-4.732,4.853,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.621,-5.272,6.154,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.878,-5.906,6.732,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.962,-5.333,6.634,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.191,-4.127,7.061,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.56,-3.518,6.603,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.747,-7.121,7.246,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.89,-7.862,7.764,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.562,-8.322,9.186,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.461,-8.814,9.45,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.212,-9.061,6.861,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.561,-8.62,5.425,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.148,-9.729,4.642,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.23,-9.266,3.185,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.326,-10.005,2.519,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.561,-8.204,10.075,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.394,-8.669,11.443,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.407,-10.184,11.471,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.2,-10.829,10.767,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.527,-8.08,12.293,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.492,-10.767,12.257,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.425,-12.219,12.372,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.28,-12.554,13.835,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.274,-11.664,14.686,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.336,-12.825,11.514,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.093,-12.307,11.846,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.137,-13.858,14.107,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.938,-14.395,15.449,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.21,-14.365,16.299,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-94.348,-13.511,17.177,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.806,-13.646,16.138,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.378,-14.307,17.44,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.671,-15.511,17.648,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-90.739,-13.614,18.242,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-95.111,-15.329,16.072,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.965,-14.206,13.463,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.963,-13.083,12.696,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-92.426,-18.97,18.216,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.187,-17.875,18.08,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.137,-18.927,17.875,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.97,-22.202,15.585,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-80.486,-8.502,0.735,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.067,-25.686,15.249,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-83.653,-13.315,18.967,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-75.666,-14.428,6.898,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-89.377,-15.776,19.661,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-76.894,-12.233,5.227,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.576,-13.716,17.612,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-93.39,-11.527,0.192,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-98.699,-10.823,3.778,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-96.454,-9.266,-0.211,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-91.633,-10.615,-2.952,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-79.615,-11.032,14.063,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-86.458,-11.815,-5.635,0.3,COLOR,0.0,0.0,1.0,1.0,SPHERE,-82.095,-21.564,-2.977,0.3

            ]
cmd.load_cgo(Points_0, "Points_0", state=1)
cmd.set("cgo_transparency", 0, "Points_0")
        

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
