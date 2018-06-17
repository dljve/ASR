# ASR


## Train on spectrogram / mfcc
- In `tensorflow/examples/speech_commands/input_data.py`, line 487, use self.mfcc_ or self.spectrogram
- In `tensorflow/examples/speech_commands/models.py`, line 50,  comment/uncomment things with mfcc / spectrogram title


## Change number of filters
- In `tensorflow/examples/speech_commands/models.py`, line 242, change first_filter_count

## Change number of steps
- In `tensorflow/examples/speech_commands/train.py`, line 381, change default


## Visualize

### Print layers of CNN:
- `python tensorflow\python\tools\inspect_checkpoint.py --file_name=D:\tmp\speech_commands_train\conv.ckpt-18000` (windows)
- `python tensorflow/python/tools/inspect_checkpoint.py --file_name=/tmp/speech_commands_train/conv.ckpt-18000`(linux)


Results in:
```
Variable (DT_FLOAT) [20,8,1,64]
Variable_1 (DT_FLOAT) [64]
Variable_2 (DT_FLOAT) [10,4,64,64]
Variable_3 (DT_FLOAT) [64]
Variable_4 (DT_FLOAT) [62720,12]
Variable_5 (DT_FLOAT) [12]
global_step (DT_INT64) []
```

### Visualize convolutional layers
- `python vis_conv_layer1.py`


### Visualize spectrogram and mfcc
- (optional) choose which file to visualize by setting the input_wav
- `python wav_to_spectrogram`



## Results
### MFCC (40coeff) + 64 filters
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 89%

### MFCC (40 coeff) + 32 filters
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 88.3%
- training time: 70 min

### MFCC (40 coeff) + 21 filters (optimal according to calculation)
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc =

### MFCC (40 coeff) + 16 filters
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 86.6%

### MFCC (40 coeff) + 16 filters v2
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 87.2%

### MFCC (40 coeff) + 8 filters
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 85.8%

### MFCC (40 coeff) + 8 filters v2
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 85.9%

### MFCC (40 coeff) + 4 filters
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 81.3%
- training time: 67 min

### MFCC (40 coeff) + 2 filters
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 77.2%

### spectrogram + 64 filters
- 15000 steps (0.001 learning_rate), 3000 steps (0.0001 learning_rate) = total 18000s
- acc = 65.8%
- training time: 120 min

### spectrogram + 64 filters + extra steps
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 76.4%
- training time: 240 min

### spectrogram + 32 filters + extra steps
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 76.0%
- training time: 180 min

### spectrogram + 16 filters + extra steps
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 76.1%
- training time: 150 min

### spectrogram + 16 filters + extra steps v2
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 75.8%

### spectrogram + 8 filters + extra steps
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 71.7%
- training time: 130 min

### spectrogram + 8 filters + extra steps v2
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 72.3%

### spectrogram + 4 filters + extra steps
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 67.3%
- training time: 130 min

### spectrogram + 2 filters + extra steps
- 30000 steps (0.001 learning_rate), 6000 steps (0.0001 learning_rate) = total 36000s
- acc = 65.3%
- training time: 125 min
