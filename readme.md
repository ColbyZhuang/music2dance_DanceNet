# Music2Dance: DanceNet
This library provides the code of DanceNet with PyTorch.
In this [paper](https://arxiv.org/abs/2002.03761v2), 
we propose a novel framework, DanceNet, to generate 
3D dance motion. The video is shown in [youtube](https://www.youtube.com/watch?v=bTHSrfEHcG8).



## Code organization

    ├── README.md             <- Top-level README.
    ├── demo.sh            <- Set CUDA_VISIBLE_DEVICES.
    ├── main.py            <- training / testing 
    ├── music_dance_pair_view.py            <- visualization of music and dance motion
    │    
    ├── data
    │   ├── read_motion_utils           <- data processing
    │   ├── motion_music_align_dance1anddance2           <- data path
    │   ├── multi_dance_prepare.py           <- dataloader
    │   ├── standrand_sps.sps             <- the data format
    │
    ├── models
    │   ├── dancenet_rhythm_chroma.py        <- models, loss
    │   ├── Optim.py                      <- optimize
    │
    ├── utils
    │   ├── hparams.py        <- hyper-parameter
    │   ├── data_utils.py     
    │   ├── logger.py      
    │   ├── misc.py     
    │   ├── osutils.py              


## Testing example 
```bash
bash demo.sh
```

## visualization 
We test the code in windows.     
Requirement:PyQt5,PyOpenGL.
```bash
python music_dance_pair_view.py
```


