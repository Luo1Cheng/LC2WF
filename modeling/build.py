from modeling.transformer import Mytransformer, Mytransformer_classify


def build_model(cfg,model=None):
    if model == "classify":
        return Mytransformer_classify(cfg)
    return Mytransformer(cfg)
