python test_multiview.py --config ../experiments/siammask_sharp/config_davis.json --resume ../experiments/siammask_sharp/SiamMask_DAVIS.pth --mask --refine --dataset ytb_vos -l ../experiments/siammask_sharp/logs/test_ytb_multiview_fast_gt_trim --visualization --K-exemplars 7 --multiview

python test_multiview.py --config ../experiments/siammask_sharp/config_davis.json --resume ../experiments/siammask_sharp/SiamMask_DAVIS.pth --mask --refine --dataset ytb_vos -l ../experiments/siammask_sharp/logs/test_ytb_multiview_original --visualization --K-exemplars 1
