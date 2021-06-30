"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import os
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm.notebook import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

a = 10
@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False  
    if classify:  #不可能執行
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)  #video / img 吧
        #print(f"this is dataset : {dataset}")
    #print(f"this is source : {source}")
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    #0.5個def
    def center_distance(xyxy1, xyxy2):
        '''Calculate the distance of the centers of the boxes.'''#計算盒子中心的距離
        #print(xyxy1)
        a, b, c, d, ex1, _ = xyxy1
        x1 = int(np.mean([a, c]))#求取均值
        y1 = int(np.mean([b, d]))
        #xyxy1 => [     52.575       179.4       79.95       246.9]

        e, f, g, h, ex2, __ = xyxy2
        x2 = int(np.mean([e, g]))
        y2 = int(np.mean([f, h]))
        #xyxy2 => [      638.4       133.2       664.8         207]
        
        dist = np.linalg.norm([x1 - x2, y1 - y2])#https://www.delftstack.com/zh-tw/howto/numpy/calculate-euclidean-distance/
        #歐幾里得距離
        return dist, x1, y1, x2, y2

    #一個def
    def detect_people_on_frame(filename,det,add_height,add_width,confidence, distance):
        coord = det
        coord = coord[coord[:, 4] >= confidence] #不確定conf變數
        coord = coord[coord[:, 5] == 0]
        coord = coord[:, :6]
        #print(f"this is coord: {coord}")
        listclass = coord[:,-1].tolist()
        toint = [int(i) for i in listclass]
        # print(f"this is img first: {img.cpu().numpy()}")
        # print(f"this is img first type: {type(img.cpu().numpy())}")
        border_size=150
        border_text_color=[255,255,255]
        img = cv2.copyMakeBorder(filename, border_size,0,0,0, cv2.BORDER_CONSTANT) 
        person =toint.count(0)
        text = "personCount: {} ".format(person) 
        cv2.putText(img ,text , (0, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,border_text_color, 2)
        colors = ['green']*len(coord[:, :4])
        lineCount = 0          
        for i in range(len(coord[:, :4])):
                for j in range(i+1, len(coord[:, :4])):
                    # Calculate distance of the centers #計算中心距離
                    dist, x1, y1, x2, y2 = center_distance(coord[i], coord[j]) #注 意
                    if dist < distance: #當距離過近 會出現紅線 =>紅線總數+1
                        lineCount +=1
                        # If dist < distance, boxes are red and a line is drawn
                        colors[i] = 'red'
                        colors[j] = 'red'
                        img = cv2.line(img, (int(x1)-add_width, int(y1)+border_size-5-add_height), (int(x2)-add_width, int(y2)+border_size-add_height), (0, 0, 255), 2) #都是以BGR表示，不是RGB

        text="redLine: {} ".format(lineCount) 
        cv2.putText(img ,text , (300, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,border_text_color, 2)            
        for i, (x1, y1, x2, y2, c1, label) in enumerate(coord):
            
            # Draw the boxes   
            if colors[i] == 'green':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)          #img = cv2.rectangle(img, (int(x1), int(y1)+border_size), (int(x2), int(y2)+border_size), color, 2) 
            img = cv2.rectangle(img, (int(x1)-add_width, int(y1)+border_size-add_height), (int(x2)-add_width, int(y2)+border_size-add_height), color, 2) #之後再看有沒有聰明一點方式轉成int
            # print(int(label))                   #用減去的方式 會使方框上移
            # print(type(label))
            text = "{}: {:.4f}".format(model.names[int(label)],c1)
            img = cv2.putText(img, text, (int(x1)-add_width, int(y1)+border_size-5-add_height), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1) 
            #img = cv2.putText(img, text, (int(x1), int(y1)+border_size-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1) 
        #Add top-border to frame to display stats


        return img
    detlist=[]
    lenframe=[]
    for path, img, im0s, vid_cap in dataset:
        # print(f"this is img first: {img}")
        # print(f"this is img first shape: {img[:, :, ::-1].shape}")
        img = torch.from_numpy(img).to(device)
        # print(f"this is img numpy: {img}")
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # print(f"this is img idk: {img}")
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # print(f"this is img 255: {img}")
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print(f"this is img hehe: {img}")
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # print(f"this is pred(model) : {pred}")
        # print(f"this is pred(model) shape : {pred.shape}")
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print(f"this is pred(non_max_suppression) : {pred}")
        # print(f"this is pred(non_max_suppression) len: {len(pred)}")
        # print(f"this is pred(non_max_suppression) shape : {pred[0].shape}")
        #print(f"this is conf_thres : {conf_thres}")
        t2 = time_synchronized()
        # print(f"this is pred after: {pred}")    
        #coordinates =pred[0].cpu().numpy()
        
        # print(f"this is pred : {pred[0].cpu().numpy()}")
        # print(f"this is pred type: {type(pred[0].cpu().numpy())}")
        # print(f"this is pred len: {len(pred)}")

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            #lenframe.append(frame)
            #print(f"this is frame: {frame}")
            detlist.append(det.cpu().numpy())
            # print(f"this is det real: {det.cpu().numpy()}")
            




            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            # print(f"this is first s: {s}") 
            get_s = s
            # print(f"this is img shape: {img.shape[2:]}")
            # print(f"this is img shape: {type(img.shape[2:])}")
            # print(f"this is img shape: {img.shape[2:].tolist()}")
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(f"this is det: {type(det[:, :4])}")
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # print(f"this is n: {n}")   
                    # print(f"this is s in for: {s}")   

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print(f"this is xywh: {xywh}")   
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        print(f"this is line : {line}")   
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # print(f"this is conf: {conf}") 
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')
    #print(f"this is all detlist: {detlist}")
    # print(f"this is all detlist len: {len(detlist)}")
    # print(f"this is save_txt: {bool(save_txt)}") 
    # print(f"this is save_crop: {bool(save_crop)}") 
    # print(f"this is view_img: {bool(view_img)}") 

    # print(f"this is get s: {get_s}")
    # print(f"this is get s type: {type(get_s)}")
    #it's time 看 這 邊
    def detect_people_on_video(filename, confidence, distance=60):
        get_wh = get_s.split("x") #[height, width]]
        cap = cv2.VideoCapture(source)    
        border_size=150
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        add_height = int(get_wh[0]) - height
        add_width = int(get_wh[1]) - width
        print(f"this is add_height: {add_height}") 
        print(f"this is width: {width}") 
        print(f"this is height: {height}") 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if os.path.exists('output.mp4'): #如果存在相同檔案，會報錯
            os.remove('output.mp4')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height+border_size)) #一定要+border_size
        vidlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #獲取影片總偵數，不用int包起來的話是 float
        i = 0 
        with tqdm(total=vidlen) as pbar:  
            while cap.isOpened():
                # Read a frame
                ret, frame = cap.read() #第一个参数ret的值为True或False，代表有没有读到图片 #第二个参数是frame，是当前截取一帧的图片。
    
                # If it's ok
                #print(f"this is frame: {frame}")
                if ret == True:           
                    
                    frame = detect_people_on_frame(frame, detlist[i],add_height,add_width,confidence, distance) # 要 改
                        
                    # Write new video
                    out.write(frame) #寫入影格
                    pbar.update(1)
                    i += 1
                else:
                    break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()       
    detect_people_on_video(source, conf_thres, distance=60)

 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
