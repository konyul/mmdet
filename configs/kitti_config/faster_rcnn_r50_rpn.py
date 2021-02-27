_base_ = ['../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_kitti.py'
]




data = dict(samples_per_gpu=2)
model = dict(rpn_head=dict(anchor_generator=dict(scales=[2,4,6,8])),
roi_head=dict(bbox_head=dict(num_classes=3)))
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
optimizer = dict(type='SGD', lr=0.02 / 8, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12 #12


evaluation = dict(interval=1, metric='mAP') #epoch단위로 화면에 값 띄워줄건지
#interval=12
# python tools/train.py configs/kitti_config/faster_rcnn_r50_rpn.py --seed 0 --work-dir work_dir/ 

# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/dist_train.sh ./configs/kitti_config/faster_rcnn_r50_rpn.py 4 --seed 0 --work-dir work_dir/ 