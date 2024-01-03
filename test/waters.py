
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


Points_0 = [
        
COLOR,1.0,0.0,0.0,1.0,SPHERE,-88.594,-14.153,14.707,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-87.376,-11.777,11.616,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-93.002,-16.763,-1.209,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-83.198,-14.78,-0.382,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-88.706,-15.047,-4.054,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-89.896,-18.541,12.036,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-84.159,-12.835,13.748,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-81.39,-10.255,6.935,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-90.706,-15.266,-2.224,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-87.669,-16.406,10.958,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-81.965,-11.672,13.293,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-88.907,-14.173,12.119,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-82.542,-13.395,8.904,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-91.386,-17.029,13.749,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-79.364,-9.225,5.156,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-89.653,-15.968,15.487,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-79.74,-12.235,6.449,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-85.209,-17.09,10.06,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-87.292,-18.551,13.482,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,-86.714,-16.38,-3.999,0.3

            ]
cmd.load_cgo(Points_0, "Points_0", state=1)
cmd.set("cgo_transparency", 0, "Points_0")
        

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()
