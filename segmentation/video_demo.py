# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import mmcv
import shutil
import json

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import numpy as np
import tqdm

cocostuff_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
    'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
    'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
    'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
    'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
    'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
    'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
    'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
    'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
    'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
    'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
    'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
    'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
    'window-blind', 'window-other', 'wood'
]

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('video', help='Video file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
    
    mmcv.mkdir_or_exist(osp.join(args.out, osp.basename(args.video)[:-len('.mp4')]))
    video = cv2.VideoCapture(args.video)

    # writer0 = cv2.VideoWriter(
    #     filename=osp.join(args.out, osp.basename(args.video)[:-len('.mp4')], osp.basename(args.video)[:-len('.mp4')] + '-background.mp4'),
    #     # some installation of opencv may not support x264 (due to its license),
    #     # you can try other format (e.g. MPEG)
    #     fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    #     fps=float(video.get(cv2.CAP_PROP_FPS)),
    #     frameSize=(1280, 720),
    #     isColor=True,
    # )

    # writer1 = cv2.VideoWriter(
    #     filename=osp.join(args.out, osp.basename(args.video)[:-len('.mp4')], osp.basename(args.video)[:-len('.mp4')] + '-foreground_mask.mp4'),
    #     # some installation of opencv may not support x264 (due to its license),
    #     # you can try other format (e.g. MPEG)
    #     fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    #     fps=float(video.get(cv2.CAP_PROP_FPS)),
    #     frameSize=(1280, 720),
    #     isColor=True,
    # )
    
    if hasattr(model, 'module'):
        model = model.module

    game_id = osp.basename(args.video)[:-len('.mp4')]
    
    frames_dir = osp.join(args.out, game_id, 'frames')
    try:
        shutil.rmtree(frames_dir)
    except: pass
    try:
        os.mkdir(frames_dir)
    except: pass
    
    semseg_dir = osp.join(args.out, game_id, 'semseg')
    try:
        shutil.rmtree(semseg_dir)
    except: pass
    try:
        os.mkdir(semseg_dir)
    except: pass

    # semseg_fg_dir = osp.join(args.out, game_id, 'semseg_fg')
    # try:
    #     shutil.rmtree(semseg_fg_dir)
    # except: pass
    # try:
    #     os.mkdir(semseg_fg_dir)
    # except: pass

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm.tqdm(range(frame_count)):
        success, frame = video.read()
        if not success: break
        
        result = inference_segmentor(model, frame)
        
        foreground_mask = result[0] == 0
        for category in [93, 131]:
            foreground_mask = foreground_mask | (result[0] == category)
        cv2.imwrite(osp.join(args.out, game_id, f'frames/{i}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # frame_foreground = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2BGRA)
        # frame_foreground[~foreground_mask] = 0
        # cv2.imwrite(osp.join(args.out, game_id, f'semseg_fg/{i}.png'), frame_foreground)
        
        foreground_mask = foreground_mask.astype(int) * 255
        foreground_mask = cv2.merge((foreground_mask,foreground_mask,foreground_mask, foreground_mask))
        cv2.imwrite(osp.join(args.out, game_id, f'semseg/{i}.png'), foreground_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        # writer1.write(foreground_mask)
        # frame_background = frame.copy()
        # frame_background[foreground_mask] = 0
        # writer0.write(frame_background)

        # if i % 20 == 0:
        #     print(f'Finished processing {i} frames')
    with open(osp.join(args.out, game_id, f'frames/max_frame.json'), 'w') as f:
        json.dump({'max_frame': frame_count}, f)        
    
    # writer0.release()
    # writer1.release()

if __name__ == '__main__':
    main()