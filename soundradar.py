import pyaudio
import numpy as np 
import time
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as mpl


FORMAT = pyaudio.paFloat32
CHANNELS = 2 
RATE = 48000
CHUNK = 2048
RECORD_SECONDS = 5
audio = pyaudio.PyAudio()

drawBuff = np.zeros((1025, 512), dtype = np.float32)


def getVolRatio(sampleL, sampleR):
    volL = np.sum(sampleL ** 2) / len(sampleL)
    volR = np.sum(sampleR ** 2) / len(sampleR)
    return 100 * ((volL - volR) / (volL + volR))


def onStreamData(in_data, frame_count, time_info, flag):
    result = np.frombuffer(in_data, dtype = np.float32)
    result = np.reshape(result, (frame_count, CHANNELS))
    left = result[:, 0]
    right = result[:, 1]

    Xdb = librosa.amplitude_to_db(abs(librosa.stft(left)))
    length = Xdb.shape[1]

    drawBuff[:, :-length] = drawBuff[:, length:]
    drawBuff[:, -length:] = Xdb

    return(None, pyaudio.paContinue)


def drawInit(): 
    ld.specshow(dataBuff, sr = RATE, x_axis = 'time', y_axis = 'hz') 
    plt.colorbar()


def onDraw(ignored): 
    ld.specshow(drawBuff, sr = RATE, x_axis = 'time', y_axis = 'hz')


print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
print("-------------------------------------------------------------")

index = int(input())
print("recording via index " + str(index))

stream = audio.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        #input_device_index = index,
        frames_per_buffer = CHUNK,
        stream_callback = onStreamData)

print ("recording started")

#data = stream.read(CHUNK)
#onStreamData(data, CHUNK, None, None)

plot = plt.figure(figsize = (15, 3))
a = anim.FuncAnimation(plot, onDraw, interval = 1000) #, init_func = drawInit)
plt.show()

stream.start_stream()
time.sleep(RECORD_SECONDS)

print ("recording stopped")

stream.stop_stream()
stream.close()
audio.terminate()
 
