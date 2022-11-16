import torch
import torchvision
from PIL import Image
import os
# import albumentations as A

class CocoDetection(torch.utils.data.Dataset):

    def __init__(self, root, annFile, img_size, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.img_size = img_size
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
    
    def convert_bbox_to_yolo_bbox(self, bbox):
        minx, miny, w, h = bbox
        centerx = minx + (w // 2)
        centery = miny + (h // 2)
        
        return torch.tensor([centerx, centery, w, h])
    
    def resize_bbox(self, bbox, img_info):
        x_scale = img_info['new_w'] / img_info['w']
        y_scale = img_info['new_h'] / img_info['h']

        x, y, w, h = bbox
               
        new_x = x * x_scale
        new_y = y * y_scale
        new_w = w * x_scale
        new_h = h * y_scale

        return [new_x, new_y, new_w, new_h]

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)


        path = coco.loadImgs(img_id)[0]['file_name']

        img_info = {}
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img_info['w'], img_info['h'] = img.size
        if self.transform is not None:
            img = self.transform(img)
            img_info['new_h'], img_info['new_w'] = img.shape[1], img.shape[2]

        if self.target_transform is not None:
            target = self.target_transform(target)

        # add extra target for yolo bounding box (minx, miny, w, h) -> (centerx, centery, w, h)
        num_objs = len(target)
        for i in range(num_objs):
            target[i]['bbox'] = self.resize_bbox(target[i]['bbox'], img_info)
            target[i]['yolo_bbox'] = self.convert_bbox_to_yolo_bbox(target[i]['bbox'])

        return img, target


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((416, 416))]) 
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = CocoDetection('coco128/train2017', 'coco128/annotations/instances_train2017.json', img_size=416, transform=transform)
    test_dataset = CocoDetection('coco128/val2017', 'coco128/annotations/instances_val2017.json', img_size=416, transform=transform)

    def collate_fn(batch):
        return tuple(zip(*batch))

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    imgs, annotations = next(iter(train_loader))
    image = imgs[0].numpy().transpose(1, 2, 0)
    annotation = annotations[0]



    import cv2

    def draw_bboxes(image, annotation):
        num_obj = len(annotation)

        for i in range(num_obj):
            x = int(annotation[i]['bbox'][0])
            y = int(annotation[i]['bbox'][1])
            w = int(annotation[i]['bbox'][2])
            h = int(annotation[i]['bbox'][3])

            centerx = int(annotation[i]['yolo_bbox'][0])
            centery = int(annotation[i]['yolo_bbox'][1])

            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # draw circle at center of bbox to confirm yolo_bbox is correct
            image = cv2.circle(image, (centerx, centery), 1, (0, 255, 0), 2)

        return image

    print(image.shape)
    print(annotation[0]['yolo_bbox'])
    print(annotation[0]['bbox'])
    #convert to BGR to show image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = draw_bboxes(image, annotation)

    # bounding box does not scale with image: Fix soon
    cv2.namedWindow('bruh', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('bruh', 800, 800)
    cv2.imshow('bruh', image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
