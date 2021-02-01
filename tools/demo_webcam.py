# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *

bbox = [0,0,0,0] #[ul_r, ul_c, lr_r, lr_c]
siammask = None
img = None
img_raw = None
state = None
tracking = False
drawing = False

def mouse_event_callback(event, x, y, flags, param):
    global bbox, siammask, img, state, tracking, drawing
    
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
        state = siamese_init(img, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        tracking = True
        drawing = False

if __name__ == '__main__':
    #global bbox, siammask, img, state, tracking
    
    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

    parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
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

    #cap = cv2.VideoCapture(args.base_path)
    cap = cv2.VideoCapture(0)

    res, img_raw = cap.read()

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback("SiamMask", mouse_event_callback)

    playing = False

    while True:
        tic = cv2.getTickCount()
        img = img_raw.copy()
        
        if drawing:
            img = cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (255,0,0), 3) 

        if playing is True:
            res, img_raw = cap.read()

            if tracking:
                state = siamese_track(state, img_raw, mask_enable=True, refine_enable=True, device=device)  # track
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr
                
                img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
                cv2.polylines(img, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

                print(state['score'])
                
        cv2.imshow("SiamMask", img)
        k = cv2.waitKey(1)

        if k == ord(' '):
            playing = not playing

        elif k == ord('s'):
            tracking = False
