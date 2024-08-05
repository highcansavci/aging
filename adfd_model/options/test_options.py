from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        # arguments for ADFD
        self.parser.add_argument('--max_steps_adfd', default=10000, type=int,
                                 help='Number of iteration')
        self.parser.add_argument('--div_opt', default='adam', type=str,
                                 help='Optimizer')
        self.parser.add_argument('--div_lr', default=0.01, type=float,
                                 help='Learning rate for optimization')
        self.parser.add_argument('--patience', default=7, type=int,
                                 help='Parameter for EarlyStopping. Defualt value is 7.')
        self.parser.add_argument('--es_delta', default=0.0001, type=float,
                                 help='Parameter for EarlyStopping. Defualt value is 0.0001.')
        self.parser.add_argument('--lpips_lambda', default=0.01, type=float,
                                 help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float,
                                 help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0.01, type=float,
                                 help='L2 loss multiplier factor')
        self.parser.add_argument('--aging_lambda', default=5, type=float,
                                 help='Aging loss multiplier factor')
        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float,
                                 help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float,
                                 help='L2 loss multiplier factor for inner image region')
        self.parser.add_argument('--lpips_lambda_aging', default=0, type=float,
                                 help='LPIPS loss multiplier factor for aging')
        self.parser.add_argument('--l2_lambda_aging', default=0, type=float,
                                 help='L2 loss multiplier factor for aging')
        self.parser.add_argument('--use_weighted_id_loss', action="store_true",
                                 help="Whether to weight id loss based on change in age (more change -> less weight)")

    def parse(self):
        opts, unknown = self.parser.parse_known_args()
        return opts
