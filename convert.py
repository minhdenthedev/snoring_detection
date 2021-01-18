# Import libraries
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt 
# import pyaudio 
import time 
import os, sys

# Define variebles
N_FFT = 1024         
HOP_SIZE = 1024      
N_MELS = 128         
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 1400
count = 1

# Load audio 
files = os.listdir('raw_data\False')
print(files)
for i in files: 
    audio, rate = librosa.load("H:/snoring_detection/raw_data/False/{}".format(i))
    print("Processing {}".format(i))
    mel = librosa.feature.melspectrogram( y=audio, sr=rate,
                                            n_fft=N_FFT,
                                            hop_length=HOP_SIZE, 
                                            n_mels=N_MELS, 
                                            htk=True, 
                                            fmin=FMIN, # higher limit ##high-pass filter freq.
                                            fmax= rate/4)
    
    # Display mel-spectrogram
    fig = plt.figure(1,frameon=False)
    fig.set_size_inches(2.24,2.24)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(librosa.power_to_db(mel**2,ref=np.max), fmin=FMIN) #power = S**2
    
    plt.savefig('H:\snoring_detection\dataset\False\img{}.png'.format(count))
    print("Processed img{}".format(count))
    count += 1 