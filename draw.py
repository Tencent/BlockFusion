import os.path
import cv2
import numpy as np
import matplotlib
import argparse
from math import ceil
cmap = matplotlib.cm.get_cmap('tab10')
color = cmap(0)

classes = [ "floor", "wall", "chair", "cabinet", "sofa", "table",  "lighting", "bed",  "stool"]

color_maps = {}
for i in range (len(classes)):

    color_maps[ classes[i]] = (np.array( cmap(i) )[:3] * 255 ).astype(np.uint8)

def get_views(panorama_height, panorama_width, window_size=64, stride=32):
    num_blocks_height = ceil((panorama_height - window_size) / stride) + 1
    num_blocks_width = ceil((panorama_width - window_size) / stride) + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = [[] for _ in range(num_blocks_height)]
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            h_start = int(i  * stride)
            h_end = h_start + window_size
            w_start = int(j  * stride)
            w_end = w_start + window_size
            views[i].append((h_start, h_end, w_start, w_end))
    return views

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=int, default=32)
    parser.add_argument('--w', type=int, default=32)
    parser.add_argument('--stride', type=int, default=24)
    parser.add_argument('--dump_path', type=str, default='./layout')
    args = parser.parse_args()

    os.makedirs(args.dump_path, exist_ok=True )

    layer_maps = {}

    for cls in classes:
        layer_maps[cls] = np.zeros((args.h, args.w), dtype=np.uint8)
    background = np.zeros( (args.h, args.w, 3), dtype=np.uint8 ) * 255

    category_maps = {
        "wall" :  background.copy(),
        "floor":  background.copy()
    }

    current_layer_id = 1

    start = [0,0]

    def clickhandler(event, x, y, flags, param):
        global layer_maps, current_layer, start
        if current_layer_id == 1: # wall
            if event == cv2.EVENT_LBUTTONDOWN:
                start = (x,y)
            elif event == cv2.EVENT_LBUTTONUP:
                dx = abs(x - start[0])
                dy = abs(y - start[1])
                if dx > dy :
                    y = start [1]
                else:
                    x = start [0]
                c = color_maps[classes [1]]
                c = (int(c[0]),int(c[1]), int(c[2]))

                print ("color", c)
                cv2.line( layer_maps[ classes [1]], start, (x, y), 1, 1)#, cv2.LINE_AA)
        else : #
            if event == cv2.EVENT_LBUTTONDOWN:
                start = (x,y)
            elif event == cv2.EVENT_LBUTTONUP:

                c = color_maps[classes[current_layer_id]]
                c = (int(c[0]),int(c[1]), int(c[2]))
                print ("color", c)
                # cv2.line(layer_maps[classes[current_layer_id]], start, (x, y), c, 1)  # , cv2.LINE_AA)
                cv2.rectangle(layer_maps[classes[current_layer_id]], start, (x, y), 1, -1)

    cv2.namedWindow("layout drawer", 0)
    cv2.resizeWindow("layout drawer", args.h, args.w)
    cv2.setMouseCallback("layout drawer", clickhandler)

    save = False

    while True:

        # obtain_background
        combine_layer = np.ones((args.h, args.w, 3), dtype=np.uint8) * 175
        combine_layer = combine_layer.reshape(-1, 3)
        for cls in classes:
            combine_layer [ layer_maps[cls].reshape(-1) >0 ] = color_maps [ cls ]
        combine_layer = combine_layer.reshape ((args.h, args.w, 3))
        cv2.imshow("layout drawer", combine_layer)

        key = cv2.waitKey(1)

        for i in range(len(classes)):
            if key == ord( str(i+1)):
                print( "editing class:" , classes [ i ] )
                current_layer_id = i


        if key == ord('r'): # remove, i.e erasor
            print( key )
            current_layer_id = -1
        elif key == ord('s'): # save
            save =True
            break
        elif key == ord('q'): # save
            break

    cv2.destroyAllWindows()

    if save:

        cv2.imwrite( os.path.join(args.dump_path, "layout.png"), combine_layer )
        conditionmaps = np.zeros((9, args.h, args.w), dtype=np.uint8)
        categories = ['wall', 'floor', 'bed', 'cabinet', 'chair', 'lighting', 'sofa', 'stool', 'table']
        for i, c in enumerate(categories):
            conditionmaps[i] = layer_maps[c]
        views = get_views(args.h, args.w, window_size=32, stride=args.stride)
        hmax = 32 + (len(views) - 1) * args.stride
        wmax = 32 + (len(views[0]) - 1) * args.stride
        conditionmaps = np.pad(conditionmaps, pad_width=((0, 0), (0, hmax - args.h), (0, wmax - args.w)), mode='constant')
        for i in range(len(views)):
            for j in range(len(views[0])):
                x_start, x_end, z_start, z_end = views[i][j]
                layoutcrop = conditionmaps[:, x_start:x_end, z_start:z_end]
                np.save(os.path.join(args.dump_path, "{}.npy".format(str(i) + '_' + str(j))), layoutcrop)