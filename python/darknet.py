import sys
import detector

def print_usage():
    print('Usage: python darknet.py detect <net.cfg> <net.weights> <pic.jpg>')
    print('Usage: python darknet.py detector test <set.data> <net.cfg> <net.weights> <pic.jpg>')
    print('Usage: python darknet.py detector demo <set.data> <net.cfg> <net.weights> <camera.index> <video.avi>')

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print_usage()
        sys.exit(0)
    if sys.argv[1] == 'detect':
        sys.argv.insert(2, './cfg/coco.data')
        sys.argv.append('.5') #thresh
        sys.argv.append('.5') #hier thresh
        sys.argv.append('.45') #nms
        detector.test_detector(*(sys.argv))
    elif sys.argv[1] == 'detector':
        detector.run_detector(*(sys.argv))
    else:
        print_usage()
