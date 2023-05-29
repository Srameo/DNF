<!-- <img src="https://github.com/Srameo/DNF/assets/7869323/ed7b8296-fe3c-48f9-bad3-f777a1b80c0b" alt="23CVPR-DNF-pipeline" width="704px"> -->


# DNF: Decouple and Feedback Network for Seeing in the Dark

This repository contains the official implementation of the following paper:
> DNF: Decouple and Feedback Network for Seeing in the Dark<br/>
> [Xin Jin](https://srameo.github.io)<sup>\*</sup>, [Ling-Hao Han](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en)<sup>\*</sup>, [Zhen Li](https://paper99.github.io/), Zhi Chai, [Chunle Guo](https://mmcheng.net/clguo/), [Chongyi Li](https://li-chongyi.github.io/)<br/>
> (\* denotes equal contribution.)<br/>
> In CVPR 2023

\[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf)\]
\[Poster (TBD)\]
\[Homepage (TBD)\]
\[Video (TBD)\]


<img src="https://github.com/Srameo/DNF/assets/7869323/01633049-930c-4149-ba04-3d89faa05b69" alt="23CVPR-DNF-example-2" width="768px">


## News

> Future work can be found in [todo.md](docs/todo.md).

- **May, 2023**: Our code is publicly available.
- **Mar, 2023**: Excited to announce that our paper was selected as **CVPR 2023 Highlight✨ (10% of accepted papers, 2.5% of submissions)!**
- **Feb, 2023**: Our paper "DNF: Decouple and Feedback Network for Seeing in the Dark" has been accepted by CVPR 2023.
- **Apr, 2022**: A single-stage version of our network has won the third place in [NTIRE 2022 Night Photography Challenge](https://nightimaging.org/challenges/2022/final-leaderboard.html).

## Dependencies and Installation


1. Clone Repo
   ```bash
   git clone https://github.com/Srameo/DNF.git CVPR23-DNF
   ```
2. Create Conda Environment and Install Dependencies
   ```bash
   conda create -n dnf python=3.7.11
   conda activate dnf
   pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
   ```
3. Download pretrained models from [Pretrained Models](#pretrained-models), and put them in the pretrained folder.


## Pretrained Models

<table>
<thead>
  <tr>
    <th> Trained on </th>
    <th> :link: Download Links </th>
    <th> Config file </th>
    <th> CFA Pattern </th>
    <th> Framework </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SID Sony</td>
    <th> [<a href="https://drive.google.com/file/d/1FHreF_UHFutkiQ0LMdWjX2fahznka0Cb/view?usp=share_link">Google Drive</a>][<a href="https://pan.baidu.com/s/1-r29zUvCS-Wa2wEYovX89g?pwd=eoiz">Baidu Cloud</a>] </th>
    <th> [<a href="configs/cvpr/sony/baseline.yaml">configs/cvpr/sony/baseline</a>] </th>
    <th> Bayer (RGGB) </th>
    <th> DNF </th>
  </tr>
  <tr>
    <td>SID Fuji</td>
    <th> [<a href="https://drive.google.com/file/d/1WfwZLBbj0EUf_QTYS8Qq5Rzk2iV8QKQ7/view?usp=share_link">Google Drive</a>][<a href="https://pan.baidu.com/s/1Sz30vAfVfF0gymNgjEUqMw?pwd=biqo">Baidu Cloud</a>]</th>
    <th> [<a href="configs/cvpr/fuji/baseline.yaml">configs/cvpr/fuji/baseline</a>] </th>
    <th> X-Trans </th>
    <th> DNF </th>
  </tr>
  <tr>
    <td>MCR</td>
    <th> [<a href="https://drive.google.com/file/d/1kFYnqJTYfYkRWcojGxgV9DpVup4uFFBR/view?usp=share_link">Google Drive</a>][<a href="https://pan.baidu.com/s/18CjvaJZ1YtrTa_YUnQo8Vg?pwd=tkbz">Baidu Cloud</a>] </th>
    <th> [<a href="configs/cvpr/mcr/baseline.yaml">configs/cvpr/mcr/baseline</a>] </th>
    <th> Bayer (RGGB) </th>
    <th> DNF </th>
  </tr>
</tbody>
</table>


## Quick Demo

### Try DNF on your own RAW **images** (with RGGB Bayer pattern) !

1. Download the pretrained **DNF** ([trained on SID Sony subset](#pretrained-models)) into `[PATH]`.
2. Remember the directory of your own images as `[DIR]`. 
   > If you would like to speed up, you could process the RAW image with `.ARW` postfix into numpy array follow the `Convert Your Own RAW Images to Numpy for Acceleration` section in [demo.md](docs/demo.md).\
   > And add `-a` option in command.
3. Try DNF on your images!
   ```bash
   bash demos/images_process.sh -p [PATH] -d [DIR] -r [RATIO]
   # [RATIO] denotes the additional digital gain you would like to add on your images.
   # If your data lies in '.npy' format, you should add '-a' arguments.
   bash demos/images_process.sh -p [PATH] -d [DIR] -r [RATIO] -a  # for data in numpy format
   
   # Let's see a simple example. 
   bash demos/images_process.sh -p pretrained/dnf_sony.pth -d dataset/sid/Sony/short_pack -r 100
   # The above command would try our pretrained DNF on the SID Sony subset with additional digital gain 100.
   ```
4. Check your results in `runs/CVPR_DEMO/image_demo/results/inference`!

### Try DNF on your own RAW **video clips** (with RGGB Bayer pattern) !

1. Download the pretrained **DNF** ([trained on SID Sony subset](#pretrained-models)) into `[PATH]`.
2. Preprocess your RAW video clip and save each frame into `[DIR]` with `.npy` format.
   > You could follow the steps in `Convert Your Own Video` section of [demo.md](docs/demo.md).
3. Try DNF on your video clip!
   ```bash
   bash demos/video_process.sh -d [DIR] -p [PATH] -r [RATIO] -s [SAVE_PATH] -f [FILE_NAME]
   # [RATIO] denotes the additional digital gain you would like to add on your images, Default: 50.
   # [SAVE_PATH] and [FILE_NAME] determine where to save the result.
   
   # Let's see a simple example. 
   bash demos/video_process.sh \
     -d dataset/campus/short_pack \
     -p pretrained/dnf_sony.pth \
     -r 50 \
     -s runs/videos -f campus
   # The above command would result in a 24 fps video postprocessed by our DNF with additional digital gain 50. 
   ```
 4. Check result in `[SAVE_PATH]/[FILE_NAME].mp4`.

### Try with data provided by us

Please refer to [demo.md](docs/demo.md) to learn how to download the provided data and how to inference.

## Training and Evaluation

Please refer to [benchmark.md](docs/benchmark.md) to learn how to benchmark DNF, how to train a new model from scratch.


## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{jincvpr23dnf,
    title={DNF: Decouple and Feedback Network for Seeing in the Dark},
    author={Jin, Xin and Han, Linghao and Li, Zhen and Chai, Zhi and Guo, Chunle and Li, Chongyi},
    journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```

## License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact `xjin[AT]mail.nankai.edu.cn` and `lhhan[AT]mail.nankai.edu.cn`.

For commercial licensing, please contact `cmm[AT]nankai.edu.cn`


## Acknowledgement

This repository borrows heavily from [BasicSR](https://github.com/XPixelGroup/BasicSR) and [Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark).

