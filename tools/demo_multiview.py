# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test_multiview import *
import copy
import time

bbox = [0,0,0,0] #[ul_r, ul_c, lr_r, lr_c]
siammask = None
img = None
img_raw = None
state = None
tracking = False
drawing = False
is_file = False
view_id = 0

def bbox_to_target(box):
    w = np.abs(bbox[2] - bbox[0])
    h = np.abs(bbox[3] - bbox[1])
    target_pos = np.array([bbox[0] + w / 2, bbox[1] + h / 2])
    target_sz = np.array([w, h])
    return (target_pos, target_sz)

def mouse_event_callback(event, x, y, flags, param):
    global bbox, siammask, img, img_raw, state, tracking, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox[0] = x
        bbox[1] = y
        bbox[2] = x
        bbox[3] = y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox[2] = x
        bbox[3] = y
        
    elif event == cv2.EVENT_LBUTTONUP:
        bbox[2] = x
        bbox[3] = y
        w = np.abs(bbox[2] - bbox[0])
        h = np.abs(bbox[3] - bbox[1])
        target_pos = np.array([bbox[0] + w / 2, bbox[1] + h / 2])
        target_sz = np.array([w, h])
        if view_id is 0:
            state = siamese_init(img_raw, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        else:
            state = siamese_init_multiview(img_raw, target_pos, target_sz, state, cfg['hp'], device=device, view_id=view_id)  # init tracker

        tracking = True
        drawing = False

if __name__ == '__main__':
    #global bbox, siammask, img, state, tracking
    
    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

    parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--base_path', default='/home/cail/data/siammask_test_recordings/2021-02-02-114800.mp4', help='datasets')
    parser.add_argument('--cpu', action='store_true', help='cpu mode')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    try:
        if args.base_path:
            vid_path = int(args.base_path)
        else:
            vid_path = 0
    except ValueError:
        vid_path = args.base_path

    cap = cv2.VideoCapture(vid_path)

    if type(vid_path) == str:
        fps = cap.get(cv2.CAP_PROP_FPS)
        is_file = True

    res, img_raw = cap.read()

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback("SiamMask", mouse_event_callback)

    playing = True

    while True:
        tic = cv2.getTickCount()
        img = img_raw.copy()
        
        if drawing:
            img = cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (255,0,0), 3) 

        if playing is True:
            res, img_raw = cap.read()

            if tracking:
                state = siamese_track_multiview(state, img_raw, mask_enable=True, refine_enable=True, device=device, debug=True)  # track
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr
                
                img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
                bbox_pts = [np.int0(location).reshape((-1, 1, 2))]
                cv2.polylines(img, bbox_pts, True, (0, 255, 0), 3)
                print(state['bbox'])
                print(state['score'])
                
        cv2.imshow("SiamMask", img)
        k = cv2.waitKey(1)

        if k == ord(' '):
            playing = not playing

        elif k == ord('s'):
            tracking = False
        elif k == ord('1'):
            view_id = 0
            print(view_id)
        elif k == ord('2'):
            view_id = 1
            print(view_id)
        elif k == ord('3'):
            view_id = 2
            print(view_id)
        elif k == ord('4'):
            view_id = 3
            print(view_id)
        elif k == ord('n'):
            if tracking:
                if view_id == 0:
                    view_id = view_id + 1

                target_pos, target_sz = bbox_to_target(state['bbox'])
                
                state = siamese_init_multiview(img_raw, target_pos, target_sz, state, cfg['hp'], device=device, view_id=view_id)  # init tracker
                
                print('saved a view: ' + str(view_id))
                
                view_id = (view_id + 1)
            
        # todo: should do a calculation to see how much leftover to sleep
        if is_file and playing:
            time.sleep(1.0/float(fps))
