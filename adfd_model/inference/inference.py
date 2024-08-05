from adfd_model.models.adfd import ADFD
from options.test_options import TestOptions
import torch.nn.functional as F
import dlib
import numpy as np
from PIL import Image


predictor = dlib.shape_predictor(
    "../pretrained_models/shape_predictor_68_face_landmarks.dat")
opts = TestOptions().parse()
model = ADFD(opts)


def check_face_availablilty(img):
    from sam_model.util.align_all_parallel import face_detector
    return face_detector(img=img, predictor=predictor)


def inference(np_arr):
    model.infer(np_arr)


if __name__ == "__main__":
    inference(np.array(Image.open("../../face.png")))
