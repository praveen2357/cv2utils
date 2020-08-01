"""
/**
 * @author praveen kumar yalal
 * @email praveen2357@gmail.com
 * @desc YOLO model utilities
 */
"""

from utils import letterbox


def load_tf_model(pb_path, pbtxt_path)
    tensorflowNet = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
    return tensorflowNet


def load_dk_model(weights_path, cfg_path, user_gpu=False):
    """
    load darknet yolo model using opencv
    return model object for detections
    weightsPath: string
    configPath: string
    user_gpu: boolean
    """
    # load our serialized model from disk
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # check if we are going to use GPU
    if use_gpu:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def image_padding(im, target_size):
    """
    image: cv image object
    target_size: integer
    """
    ih, iw = target_size
    nh, nw = im.shape
    im_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    im_paded[dh:nh+dh, dw:nw+dw, :] = im_resized
    im_paded = im_paded.astype(np.uint8)
    return im_paded, dw, dh


def resize(im, bboxes, target_size):
    """
    resize the image and respective bbox
    im: opencv ndarray
    bbox: x,y,width,height
    """
    h, w, d = im.shape
    im_ltrbox = letterbox(im, target_size, target_size)
    h_l, w_l, _ = im_ltrbox.shape
    im_tr = image_padding(im_ltrbox, target_size)
    # apply resize + padding to the bboxes
    bboxes_tr = np.array(bboxes,dtype=np.float32)
    bboxes_tr[:,0] = bboxes_tr[:,0] * ((w_l*1.0)/w) + x_t
    bboxes_tr[:,1] = bboxes_tr[:,1] * ((h_l*1.0)/h) + y_t
    bboxes_tr[:,2] = bboxes_tr[:,2] * ((w_l*1.0)/w)
    bboxes_tr[:,3] = bboxes_tr[:,3] * ((h_l*1.0)/h)
    return im_tr, bboxes_tr


def draw_bboxes(frame, bboxes, num_classes):
    """
    frame: numpy array(image)
    bboxes: list
    num_classes: integer
    """
    np.random.seed(42)
    # initialize a list of colors to represent each possible class label
    COLORS = np.random.randint(0, 255, size=(num_classes, 3), dtype="uint8")
    for bbox in bboxes:
        cx, cy, w, h, confidence, class_id = bbox
        # draw a bounding box rectangle and label on the frame
        color = [int(c) for c in COLORS[class_id]]
        x1, y1 = cx-w//2, cy-h//2
        x2, y2 = cx + w//2, cy + h//2
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        text = "%d" % (confidence * 100)
        # text="class"
        cv2.putText(frame, text, (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 2)


def process_layer_output(outputs, input_size, c_th):
    """
    outputs: numpy array
    input_size: integer
    c_th: float
    """
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    start_time = time.process_time()
    # loop over each of the detections
    for detection in outputs:
        # extract the class ID and confidence (i.e., probability)
        # of the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > c_th:
            # scale the bounding box coordinates back relative to
            # the size of the image, keeping in mind that YOLO
            # actually returns the center (x, y)-coordinates of
            # the bounding box followed by the boxes' width and
            # height
            box = detection[0:4] * input_size
            (centerX, centerY, width, height) = box.astype("int")
            # update our list of bounding box coordinates,
            # confidences, and class IDs
            boxes.append([centerX, centerY, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
    # print(time.process_time() - start_time, "seconds")
    return boxes, confidences, classIDs


def image_prediction(model, target_size, im, c_th=0.5, iou_th=0.5, show_output=False):
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    im_ltrbox = letterbox(im.copy(), target_size, target_size)
    frame = image_padding(im_ltrbox, target_size)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (target_size, target_size), [0,0,0], swapRB=True, crop=False)
    # blobb = blob[0, 0, :, :]
    # cv2.imshow('blob', blobb)
    # cv2.waitKey(2000)
    model.setInput(blob)
    # get the output layers
    ln = model.getLayerNames()
    ln_output = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    # run forward pass to get output of the output layers
    layerOutputs = model.forward(ln_output)
    layerOutputs = np.vstack(layerOutputs)
    boxes, confidences, classIDs = process_layer_output(layerOutputs, input_size, c_th)
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, c_th, iou_th)
    # ensure at least one detection exists
    pred_bboxes = []
    if len(idxs) > 0 and len(boxes)>0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (int(boxes[i][0]), int(boxes[i][1]))
        (w, h) = (int(boxes[i][2]), int(boxes[i][3]))
        pred_bboxes.append([x,y,w,h,confidences[i], classIDs[i]])
    return frame, pred_bboxes


def video_prediction(path):
    """
    process the video for detections
    """
    pass


