import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from bimvee.exportIitYarp import encodeEvents24Bit
import argparse


# original version from Massi
def frame2events(input_path, output_path):
    numBins = 10
    step = 0.5 / (numBins)
    dirList = os.listdir(input_path)
    prev_ts = None
    events = []
    ev_img_list = sorted([x for x in dirList if x.__contains__('ec') and os.path.splitext(x)[-1] == '.png'])
    frame_counter = 0
    for file in tqdm(ev_img_list):
        frame_counter += 1
        if frame_counter == 456:
            print(frame_counter)
        with open(os.path.join(input_path, file.split('.')[0] + '.json'), 'r') as jsonFile:
            metadata = json.load(jsonFile)
        timestamp = metadata['timestamp']
        if prev_ts is None:
            prev_ts = timestamp
            continue
        image = plt.imread(os.path.join(input_path, file))
        vCount = np.round(image / step - numBins).astype(int)
        vIndices = vCount.nonzero()
        if vIndices[0].shape[0] > 10000:
            print(vIndices[0].shape)
        for y, x in zip(vIndices[0], vIndices[1]):
            num_events = vCount[y, x]
            for v in range(abs(num_events)):
                polarity = 1 if num_events > 0 else 0
                # ts_noise = (np.random.rand() - 0.5) / 250
                ts_noise = (np.random.rand() - 0.5) / 2000
                # ts_noise = 0

                ts = prev_ts + v * ((timestamp - prev_ts) / abs(num_events)) + ts_noise
                events.append([x, y, ts, polarity])
        prev_ts = timestamp

    events = np.array(events)
    events = events[events[:, 2].argsort()]
    prev_ts = events[0, 2]

    # dataDict = {'x': events[:, 0], 'y': events[:, 1], 'ts': events[:, 2], 'pol': events[:, 3].astype(bool)}
    encodedData = np.array(
        encodeEvents24Bit(events[:, 2] - events[0, 2],
                          events[:, 0],
                          events[:, 1],
                          events[:, 3].astype(bool))).astype(np.uint32)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'binaryevents.log'), 'wb') as f:
        encodedData.tofile(f)


# load event file from .npy
def frame2events2(input_path, output_path):
    numBins = 10
    step = 0.5 / (numBins)  # 0.05
    dirList = os.listdir(input_path)
    prev_ts = None
    events = []
    ev_img_list = sorted([x for x in dirList if x.__contains__('ec_2') and os.path.splitext(x)[-1] == '.npy'])
    frame_counter = 0
    for file in tqdm(ev_img_list):
        frame_counter += 1
        if frame_counter == 456:
            print(frame_counter)
        with open(os.path.join(input_path, file.split('.')[0] + '.json'), 'r') as jsonFile:
            metadata = json.load(jsonFile)
        timestamp = metadata['timestamp']
        if prev_ts is None:
            prev_ts = timestamp
            continue
        image = np.load(os.path.join(input_path, file))
        vCount = np.round(image / step - numBins).astype(int)  # (image/0.05 - 10)
        vIndices = vCount.nonzero()
        if vIndices[0].shape[0] > 10000:
            print(vIndices[0].shape)
        for y, x in zip(vIndices[0], vIndices[1]):
            num_events = vCount[y, x]
            for v in range(abs(num_events)):
                polarity = 1 if num_events > 0 else 0
                ts_noise = (np.random.rand() - 0.5) / 2000
                ts = prev_ts + v * ((timestamp - prev_ts) / abs(num_events)) + ts_noise

                # simulate event packet
                # ts_noise = (np.random.rand() - 0.5) / 2500
                # ts = prev_ts + 0 * ((timestamp - prev_ts) / abs(num_events)) + ts_noise
                # ts = prev_ts

                events.append([x, y, ts, polarity])
        prev_ts = timestamp

    events = np.array(events)
    events = events[events[:, 2].argsort()]
    prev_ts = events[0, 2]

    # dataDict = {'x': events[:, 0], 'y': events[:, 1], 'ts': events[:, 2], 'pol': events[:, 3].astype(bool)}
    encodedData = np.array(
        encodeEvents24Bit(events[:, 2] - events[0, 2],
                          events[:, 0],
                          events[:, 1],
                          events[:, 3].astype(bool))).astype(np.uint32)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'binaryevents.log'), 'wb') as f:
        encodedData.tofile(f)


# load event file from .npy
# Introduce post-processing refractory period, save the lasted event timestamp into TS,
# which controls time gap between consecutive events
def frame2events_refractory(input_path, output_path):
    numBins = 10
    step = 0.5 / (numBins)  # 0.05
    dirList = os.listdir(input_path)
    prev_ts = None
    events = []
    ev_img_list = sorted([x for x in dirList if x.__contains__('ec_2') and os.path.splitext(x)[-1] == '.npy'])
    frame_counter = 0

    # define SAE to keep track the last timestamp of the last triggered event
    img_height = 480
    img_width = 640
    TS_p = np.zeros((img_height, img_width))
    TS_n = np.zeros((img_height, img_width))

    refractory_period = 0.005

    for file in tqdm(ev_img_list):
        frame_counter += 1
        if frame_counter == 456:
            print(frame_counter)
        with open(os.path.join(input_path, file.split('.')[0] + '.json'), 'r') as jsonFile:
            metadata = json.load(jsonFile)
        timestamp = metadata['timestamp']
        if prev_ts is None:
            prev_ts = timestamp
            continue
        image = np.load(os.path.join(input_path, file))
        vCount = np.round(image / step - numBins).astype(int)  # (image/0.05 - 10)
        vIndices = vCount.nonzero()
        if vIndices[0].shape[0] > 10000:
            print(vIndices[0].shape)
        for y, x in zip(vIndices[0], vIndices[1]):
            num_events = vCount[y, x]
            for v in range(abs(num_events)):
                if num_events > 0 and prev_ts - TS_p[y, x] > refractory_period:
                    polarity = 1
                    ts_noise = (np.random.rand() - 0.5) / 2000
                    ts = prev_ts + ts_noise
                    TS_p[y, x] = ts
                    events.append([x, y, ts, polarity])
                elif num_events < 0 and prev_ts - TS_n[y, x] > refractory_period:
                    polarity = 0
                    ts_noise = (np.random.rand() - 0.5) / 2000
                    ts = prev_ts + ts_noise
                    TS_n[y, x] = ts
                    events.append([x, y, ts, polarity])

        prev_ts = timestamp

    events = np.array(events)
    events = events[events[:, 2].argsort()]
    prev_ts = events[0, 2]

    # dataDict = {'x': events[:, 0], 'y': events[:, 1], 'ts': events[:, 2], 'pol': events[:, 3].astype(bool)}
    encodedData = np.array(
        encodeEvents24Bit(events[:, 2] - events[0, 2],
                          events[:, 0],
                          events[:, 1],
                          events[:, 3].astype(bool))).astype(np.uint32)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'binaryevents.log'), 'wb') as f:
        encodedData.tofile(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract binary events from sequence of frames')
    parser.add_argument('--input', '-i', dest='input_path', type=str, required=False,
                        help='Path to input file',
                        default='/home/cappe/Users/matte/Desktop/uni5/Tesi/IIT/code/photorealistic_test/')
    parser.add_argument('--output', '-o', dest='output_path', type=str, required=False,
                        help='Path to output file',
                        default='/home/cappe/Users/matte/Desktop/uni5/Tesi/IIT/code/photorealistic_test/')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path if args.output_path is not None else input_path

    frame2events(input_path, output_path)
