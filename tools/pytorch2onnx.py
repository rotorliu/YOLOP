import argparse
import os, sys
import os.path as osp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import onnx
import onnxruntime as rt
import torch

import cv2
from lib.config import cfg
from lib.models import get_net
import torchvision.transforms as transforms

from thop import profile

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 do_simplify=False):
    
    # read image
    one_img = cv2.imread(input_img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    one_img = cv2.resize(one_img, input_shape[2:])
    one_img = transform(one_img)
    one_img = one_img.unsqueeze(0) if one_img.ndimension() == 3 else one_img
    one_img = one_img.float()

    output_names = ['det1', 'det2', 'det3', 'seg']
    input_name = 'input'

    torch.onnx.export(
        model, one_img,
        output_file,
        input_names=[input_name],
        output_names=output_names,
        export_params=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version)

    if do_simplify:
        import onnxsim

        input_dic = {input_name: one_img.detach().cpu().numpy()}
        onnx_model = onnx.load(output_file)
        model_simp, check = onnxsim.simplify(onnx_model, input_data=input_dic)
        assert check, "Simplified ONNX model could not be validated"
            
        onnx.save(model_simp, output_file)
        
    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_det_pred, pytorch_da_seg_pred = model(one_img)
        pytorch_det1_pred = pytorch_det_pred[0].detach().cpu().numpy()
        print(f'pytorch_det1_pred shape: {pytorch_det1_pred.shape}')
        pytorch_det2_pred = pytorch_det_pred[1].detach().cpu().numpy()
        print(f'pytorch_det2_pred shape: {pytorch_det2_pred.shape}')
        pytorch_det3_pred = pytorch_det_pred[2].detach().cpu().numpy()
        print(f'pytorch_det3_pred shape: {pytorch_det3_pred.shape}')
        pytorch_da_seg_pred = pytorch_da_seg_pred.detach().cpu().numpy()
        print(f'pytorch_da_seg_pred shape: {pytorch_da_seg_pred.shape}')

        # get onnx output
        sess = rt.InferenceSession(output_file)
        det1_pred, det2_pred, det3_pred, da_seg_pred= sess.run(
            output_names, {input_name: one_img.detach().cpu().numpy()})
        # only compare a part of result
        assert np.allclose(
            pytorch_det1_pred, det1_pred, rtol=1.e-2, atol=1.e-6
        ) and np.allclose(
            pytorch_det2_pred, det2_pred, rtol=1.e-2, atol=1.e-6
        ) and np.allclose(
            pytorch_det3_pred, det3_pred, rtol=1.e-2, atol=1.e-6
        ) and np.allclose(
            pytorch_da_seg_pred, da_seg_pred, rtol=1.e-3
        ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')
        
        flops, params = profile(model, inputs=(one_img, ))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert models to ONNX')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[416, 416],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../tests/data/color.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # build the model
    model = get_net(cfg)
    model.eval()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # conver model to onnx file
    pytorch2onnx(
        model,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        do_simplify=args.simplify)
