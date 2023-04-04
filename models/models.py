
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'CTrGAN':
        assert(opt.dataset_mode == 'unaligned_sequence')
        from .CTrGAN_model import CTrGANModel
        model = CTrGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
