
from pymol.cgo import *
from pymol import cmd
import numpy as np
from chempy.brick import Brick
from collections import defaultdict
positions_viewport_callbacks = defaultdict(lambda: defaultdict(lambda: ViewportCallback([],0,0)))


cone_test = [
        
COLOR,1.0,0.0,0.0,1.0,SPHERE,1.0,0.0,2.220446049250313e-16,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,0.0,-0.13052619222005135,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,0.10204948636430994,-0.08138174972465471,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,0.12725362806613827,0.029044810198567503,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,0.056633192333213116,0.11760003563611363,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,-0.05663319233321309,0.11760003563611364,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,-0.12725362806613827,0.029044810198567538,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,-0.10204948636430995,-0.08138174972465467,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9914448613738104,-3.1969696701435666e-17,-0.13052619222005135,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890684,0.0,-0.2588190451025205,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,0.10527118956908339,-0.23644296300480305,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,0.19234003410293857,-0.17318374458667016,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,0.2461515393860416,-0.07997948340457471,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,0.25740120729276555,0.027053957048968166,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,0.2241438680420134,0.12940952255126054,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,0.15213001772368218,0.20938900595583548,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890682,0.053811505283103106,0.2531632279912455,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890682,-0.05381150528310293,0.2531632279912455,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,-0.15213001772368212,0.20938900595583554,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,-0.22414386804201328,0.1294095225512607,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,-0.25740120729276555,0.0270539570489684,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,-0.24615153938604162,-0.07997948340457466,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,-0.19234003410293868,-0.17318374458667005,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890683,-0.1052711895690836,-0.23644296300480294,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9659258262890684,-6.33923830286353e-17,-0.2588190451025205,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,0.0,-0.38268343236508956,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,0.1032467544273887,-0.368492492605937,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,0.19883616940137103,-0.3269721504111862,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,0.27967881530580263,-0.26120177961849506,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,0.3397789629244483,-0.17605926775013167,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,0.3746792592483138,-0.07785924541029828,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,0.3817913089572901,0.026115240979269733,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,0.36058764385992476,0.12815287941468329,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,0.31264084279617615,0.22068600504432312,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112866,0.24150690065892264,0.2968518592511215,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112866,0.15246149652035232,0.35100156906415814,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112866,0.052108720649804416,0.37911909822503936,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112866,-0.052108720649804326,0.37911909822503936,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112866,-0.1524614965203521,0.3510015690641583,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112866,-0.24150690065892255,0.29685185925112156,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,-0.3126408427961761,0.22068600504432317,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,-0.36058764385992476,0.12815287941468337,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,-0.3817913089572901,0.026115240979269913,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,-0.3746792592483139,-0.07785924541029801,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112867,-0.3397789629244485,-0.17605926775013145,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,-0.27967881530580285,-0.26120177961849483,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,-0.19883616940137103,-0.3269721504111862,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,-0.10324675442738875,-0.368492492605937,0.3,COLOR,1.0,0.0,0.0,1.0,SPHERE,0.9238795325112868,-9.373040810652597e-17,-0.38268343236508956,0.3

            ]
cmd.load_cgo(cone_test, "cone_test", state=1)
cmd.set("cgo_transparency", 0, "cone_test")
        

for x in positions_viewport_callbacks:
    for y in positions_viewport_callbacks[x]:
        positions_viewport_callbacks[x][y].load()