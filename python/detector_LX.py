import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

#import darknet as dn
#import time

"""Detector functions with different imread methods"""

import ctypes
from darknet_libwrapper import *
#from scipy.misc import imread
import cv2

def array_to_image(arr):
	arr = arr.transpose(2,0,1)
	c = arr.shape[0]
	h = arr.shape[1]
	w = arr.shape[2]
	arr = (arr/255.0).flatten()
	data = c_array(ctypes.c_float, arr)
	im = IMAGE(w,h,c,data)
	return im

def _detector_img(net, meta, image, thresh=.5, hier=.5, nms=.45):
	cuda_set_device(0)
	num = ctypes.c_int(0)
	num_ptr = ctypes.pointer(num)
	network_predict_image(net, image)
	dets = get_network_boxes(net, image.w, image.h, thresh, hier, None, 0, num_ptr)
	num = num_ptr[0]
	if (nms):
		do_nms_sort(dets, num, meta.classes, nms)

	res = []
	for j in range(num):
		for i in range(meta.classes):
			if dets[j].prob[i] > 0:
				b = dets[j].bbox
				# Notice: in Python3, mata.names[i] is bytes array from c_char_p instead of string
				res = sorted(res, key=lambda x: -x[1])
	free_detections(dets, num)
	return res

def _detector(net, meta, image, thresh=.5, hier=.5, nms=.45):
    cuda_set_device(0)
    num = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num)
    network_predict_image(net, image)
    dets = get_network_boxes(net, image.w, image.h, thresh, hier, None, 0, num_ptr)
    num = num_ptr[0]
    if (nms):
         do_nms_sort(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                # Notice: in Python3, mata.names[i] is bytes array from c_char_p instead of string
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                # Begins from yolov3, bbox coords has been changed to % of image's width and height
                # via network_predict(), remember_network() and avg_predictions()
                # TODO: solve this workaround for coords via get_network_boxes()
                dets[j].bbox.x = b.x / image.w
                dets[j].bbox.w = b.w / image.w
                dets[j].bbox.y = b.y / image.h
                dets[j].bbox.h = b.h / image.h
    #res = sorted(res, key=lambda x: -x[1])
    #draw_detections(image, dets, num, thresh, meta.names, load_alphabet(), meta.classes)
    free_detections(dets, num)
    return res



if __name__ == "__main__":
	# Darknet
	#net = load_network("cfg/yolov3.cfg", "yolov3.weights", 0)
	#meta = get_metadata("cfg/coco.data")
	net = load_network("hanshaoxing/detect_car_yolov3.cfg", "hanshaoxing/detect_car_yolov3_final.weights", 0)
	meta = get_metadata("hanshaoxing/detect_car_det.data")
	capture = cv2.VideoCapture('../../roadsegmention/othertool/asm.mp4')
	print(capture.get(cv2.CAP_PROP_FPS))
	cv2.namedWindow('frame', 0)
	cv2.resizeWindow('frame',640,480)
	# these 2 lines can be removed if you dont have a 1080p camera.
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	skip = 0
	while True:
		
		ret, frame = capture.read()
		#print(frame.shape,type(frame))
		
		skip += 1
		im = array_to_image(frame)
		rgbgr_image(im)
		if skip == 4:

			result = _detector(net, meta, im)
			#print(len(result))
			#print ('OpenCV:\n', result)
			skip = 0
			for i in result:
				cv2.circle(frame,(int(i[2][0]),int(i[2][1])),9,(0,255,255),-1)
				#cv2.rectangle(frame,(int(i[2][0]),int(i[2][1])),int((i[2][2]),int(i[2][3])),(0,255,0),-1)	
		
		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	capture.release()
	cv2.destroyAllWindows()
