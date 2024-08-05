from adfd_model.models.adfd import ADFD
from options.test_options import TestOptions
from adfd_model.config.config import *
import torch.nn.functional as F
import dlib
import numpy as np
from PIL import Image


predictor = dlib.shape_predictor(PREDICTOR_PATH)
opts = TestOptions().parse()
model = ADFD(opts)


def check_face_availablilty(img):
    from sam_model.util.align_all_parallel import face_detector
    return face_detector(img=img, predictor=predictor)


def inference(np_arr):
    return model.infer(np_arr)


if __name__ == "__main__":
    inference(np.array(Image.open("../../face.png")))
