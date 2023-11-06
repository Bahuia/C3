import os


class PredictorTrainer:
    def __init__(self, args):
        self.args = args
        self.task_num = self.args.task_num
        self.n_gpu = len(','.split(self.args.cuda_visible_devices))

        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.cuda_visible_devices

