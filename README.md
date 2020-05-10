# WaveGAN
Implementation of the paper https://arxiv.org/pdf/1802.04208.pdf



Example of piano generated with this WaveGAN implementation: [piano](https://soundcloud.com/max-holmberg-2/generated-piano-with-wavegan/s-e8zHof7Ejbs) which was trained for ~100k update steps.




In order to generate the dataset files required for training run
```
python dataset.py -create_piano_wav -path "dataset/piano/train" -output_path "piano.wav"
```
```
python dataset.py -create_piano_npy -path "piano.wav" -output_path "piano.npy"
```
```
python dataset.py -create_sc09_npy -path "dataset/sc09-spoken-numbers/sc09/train" -output_path "sc09.npy"
```

To train the model (on for example the piano dataset)

```
python run.py -train -dataset piano.npy -epochs 100
```

To continue the training and specify which logging step it should start from in tensorboard (logs to tensorboard every 10th update step, can be changed in hyperparams)
```
python run.py -train -continue -initial_log_step 5 -dataset piano.npy -epochs 100
```
