import pyaudio
import numpy as np
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import matplotlib.animation as anim

FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 48000
CHUNK = 4800 * 2
DRAW_INTERVAL = 120
RECORD_SECONDS = 5
audio = pyaudio.PyAudio()

drawBuff = np.zeros(48000, dtype=np.float32)


def on_stream_data(in_data, frame_count, time_info, flag):
    result = np.frombuffer(in_data, dtype=np.float32)
    result = np.reshape(result, (frame_count, CHANNELS))
    left = result[:, 0]
    # right = result[:, 1]

    length = left.size

    drawBuff[:-length] = drawBuff[length:]
    drawBuff[-length:] = left

    return None, pyaudio.paContinue


def draw_init():
    xdb = librosa.amplitude_to_db(abs(librosa.stft(drawBuff)))
    ld.specshow(xdb, sr=RATE, x_axis='time', y_axis='hz')
    plt.colorbar()


def on_draw(ignored):
    plt.cla()
    # xdb = librosa.amplitude_to_db(abs(librosa.stft(drawBuff)))
    xdb = librosa.amplitude_to_db(abs(librosa.stft(drawBuff, n_fft=512, hop_length=512)))
    # print(Xdb.shape)
    ld.specshow(xdb, sr=RATE, x_axis='time', y_axis='hz')


# print("----------------------record device list---------------------")
# info = audio.get_host_api_info_by_index(0)
# numdev = info.get('deviceCount')
# for i in range(0, numdev):
#     if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#         print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
# print("-------------------------------------------------------------")
#
# index = int(input())
# print("recording via index " + str(index))

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    # input_device_index = index,
    frames_per_buffer=CHUNK,
    stream_callback=on_stream_data)

print("recording started")

# data = stream.read(CHUNK)
# on_stream_data(data, CHUNK, None, None)

plot = plt.figure(figsize=(8, 4))
a = anim.FuncAnimation(plot, on_draw, interval=DRAW_INTERVAL)
# a = anim.FuncAnimation(plot, on_draw, interval = DRAW_INTERVAL, init_func = drawInit)
plt.show()

stream.start_stream()
# time.sleep(RECORD_SECONDS)

print("recording stopped")

stream.stop_stream()
stream.close()
audio.terminate()
