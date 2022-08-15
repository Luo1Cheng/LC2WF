from modeling.LineNet import LineNet,LineNetGlobal
# from modeling.LineNetV2 import LineNet,LineNetGlobal
from modeling.transformer import Mytransformer,Mytransformer_classify,Mytransformer_onlyOneTrans,Mytransformer_classify_new
from modeling.pointModel import PointNet
from modeling.LineClassify import ClassifyNetGlobal

def build_model(cfg,model=None):
    if model=="classify":
        # return Mytransformer_classify_new(cfg)
        return Mytransformer_classify(cfg)
        return ClassifyNetGlobal(cfg)
    # return Mytransformer_onlyOneTrans(cfg)
    return Mytransformer(cfg)
    return LineNetGlobal(cfg)