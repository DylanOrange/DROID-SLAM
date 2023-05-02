import cv2
import torch
import torchvision
import numpy as np
import random 

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class Segmenter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self,mean,div):
        
        self.mean = mean
        self.div = div
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().cuda()

    # @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def get_prediction(self, img, threshold):

        pred = self.model([img])
        pred_score = list(pred[0]['scores'].detach())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
        preds_count = len(pred_t)
        masks = (pred[0]['masks']>0.5).squeeze(1).detach()
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred[0]['labels']]
        pred_boxes =[[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in pred[0]['boxes'].detach()]
        masks = masks[:preds_count]
        pred_boxes = pred_boxes[:preds_count]
        pred_class = pred_class[:preds_count]
        return masks, pred_boxes, pred_class

    def instance_segmentation_api(self, img, tstamp, threshold=0.6, rect_th=1, text_size=0.6, text_th=1):

        input = img / 255.0 #1,1,3,240,808

        masks, boxes, pred_cls = self.get_prediction(input, threshold)

        mask_list, bbox_list = [], []
        for i in range(len(pred_cls)):
            if pred_cls[i] =='car' or pred_cls[i] == 'truck' or  pred_cls[i] == 'bus':
                mask_list.append(masks[i].float())
                bbox_list.append(torch.as_tensor(boxes[i],dtype=torch.int))
        masks = torch.stack(mask_list)#11,240,808, float
        bboxes = torch.stack(bbox_list)#11,4, int

        #mask: 11*240*808 bool, boxes:, pred_cls:list, len:11, 

        #visualization
        # img = img.permute(1,2,0).cpu().numpy()
        # for i in range(len(masks)):
        #     rgb_mask, color = self.random_colour_masks(masks[i])
        #     img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        #     cv2.rectangle(img, boxes[i][0] , boxes[i][1] ,color=color, thickness=rect_th)
        #     cv2.putText(img,pred_cls[i], (boxes[i][0][0], boxes[i][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, color,thickness=text_th, lineType=cv2.LINE_AA)
        # cv2.imwrite('./result/segmentation/'+str(tstamp)+'.png',img)

        return masks, bboxes

    @staticmethod
    def random_colour_masks(image):
        colours = np.random.uniform(0, 255, size=(100, 3))
        # colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        color = colours[random.randrange(0,10)]
        r[image == 1], g[image == 1], b[image == 1] = color
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask, color

    # @staticmethod
    # def visualization():
    #     img = img.permute(1,2,0).cpu().numpy()
    #     for i in range(len(masks)):
    #         rgb_mask, color = self.random_colour_masks(masks[i])
    #         img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    #         cv2.rectangle(img, boxes[i][0] , boxes[i][1] ,color=color, thickness=rect_th)
    #         cv2.putText(img,pred_cls[i], (boxes[i][0][0], boxes[i][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, text_size, color,thickness=text_th, lineType=cv2.LINE_AA)
    #     cv2.imwrite('./result/segmentation/'+str(tstamp)+'.png',img)