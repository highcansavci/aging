import torch
import dlib
from sam_model.config.config import *
from argparse import Namespace
import numpy as np
from sam_model.models.psp import pSp
from sam_model.util.common import tensor2im
from PIL import Image


model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location=DEVICE)
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = pSp(opts)
predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
net.eval()
net.cuda()
img_transforms = EXPERIMENT_ARGS['transform']


def run_alignment(img):
    from sam_model.util.align_all_parallel import align_face_numpy
    aligned_image = align_face_numpy(img_arr=img, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(),
                       randomize_noise=False, resize=False)
    return result_batch


def check_face_availablilty(img):
    from sam_model.util.align_all_parallel import face_detector
    return face_detector(img=img, predictor=predictor)


def inference(img):
    aligned_image = run_alignment(img)
    aligned_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    input_image = img_transforms(aligned_image)
    results = None
    for age_transformer in AGE_TRANSFORMERS:
        print(f"Running on target age: {age_transformer.target_age}")
        with torch.no_grad():
            input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
            input_image_age = torch.stack(input_image_age)
            result_tensor = run_on_batch(input_image_age, net)[0]
            result_image = tensor2im(result_tensor)
            # Initialize results with the first result_image shape
            if results is None:
                results = result_image
            else:
                # Concatenate the results along the height
                results = np.concatenate((results, result_image), axis=0)
    return results
