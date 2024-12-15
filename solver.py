# TODO:
    # 1. Load COCO
    # 2. Write Solver class
    # 3. Write Prediction script:
            # 1. Write nms

class Solver:
    def __init__(self, model, loss_fn, traindata, valdata=None):
        self.model = model
        self.loss_fn = loss_fn
        self.trainData = traindata
        self.valData = valdata
        self.epoch = 0
        self.history = {'train_loss': [], 'val_loss': []}
        