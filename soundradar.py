import pyaudio
import wave
import numpy as np 
import time
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import matplotlib as mpl

FORMAT = pyaudio.paFloat32
CHANNELS = 2 
RATE = 48000
CHUNK = 11024 
RECORD_SECONDS = .5
WAVE_OUTPUT_FILENAME = "recordedFile.wav"
audio = pyaudio.PyAudio()

def getVolRatio(sampleL, sampleR):
    volL = np.sum(sampleL ** 2) / len(sampleL)
    volR = np.sum(sampleR ** 2) / len(sampleR)
    return 100 * ((volL - volR) / (volL + volR))

def onStreamData(in_data, frame_count, time_info, flag):
    result = np.fromstring(in_data, dtype = np.float32)
    result = np.reshape(result, (frame_count, CHANNELS))
    left = result[:, 0]
    right = result[:, 1]

    volumeRatio = getVolRatio(left, right)

    X = librosa.stft(left)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize = (15, 3))
    ld.specshow(Xdb, sr = RATE, x_axis = 'time', y_axis = 'hz') 
    plt.colorbar()

    #print("%.2f" % volumeRatio, flush=True)
    
    return(None, pyaudio.paContinue)

print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

print("-------------------------------------------------------------")

#index = int(input())
#print("recording via index " + str(index))

stream = audio.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        #output = True,
        input = True,
        #input_device_index = index,
        frames_per_buffer = CHUNK)
        #stream_callback = onStreamData)

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    onStreamData(data, CHUNK, None, None)
    plt.show()    

print ("recording started")

#stream.start_stream()
time.sleep(RECORD_SECONDS)

print ("recording stopped")

stream.stop_stream()
stream.close()
audio.terminate()
 
