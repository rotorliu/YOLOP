import argparse
import os.path as osp

import numpy as np
import onnx
import onnxruntime as rt
import torch

import cv2
from lib.config import cfg
from lib.models import get_net
import torchvision.transforms as transforms

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
    model.cuda().eval()
    # read image
    one_img = cv2.imread(input_img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    one_img = transform(one_img)
    one_img = cv2.imresize(one_img, input_shape[2:]).transpose(2, 0, 1)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().cuda()

    output_names = ['det', 'da_seg', 'll_seg']
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

        input_dic = {'input': one_img.detach().cpu().numpy()}
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
        pytorch_det_pred, pytorch_da_seg_pred, pytorch_ll_seg_pred = model(one_img)
        pytorch_det_pred, pytorch_da_seg_pred, pytorch_ll_seg_pred = pytorch_det_pred.numpy(), pytorch_da_seg_pred.numpy(), pytorch_ll_seg_pred.numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        det_pred, da_seg_pred, ll_seg_pred = sess.run(
            None, {net_feed_input[0]: one_img.detach().cpu().numpy()})
        # only compare a part of result
        assert np.allclose(
            pytorch_det_pred, det_pred
        ) and np.allclose(
            pytorch_da_seg_pred, da_seg_pred
        ) and np.allclose(
            pytorch_ll_seg_pred, ll_seg_pred
        ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
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
        default=[384, 768],
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

    assert len(args.mean) == 3
    assert len(args.std) == 3

    # build the model
    model = get_net(cfg)
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
