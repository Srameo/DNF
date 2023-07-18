# Training and Evaluation

- [Training and Evaluation](#training-and-evaluation)
  - [Data Preparation](#data-preparation)
  - [Pretrained Models](#pretrained-models)
  - [Evaluation](#evaluation)
  - [Training](#training)

<b style='color:red'>Attention!</b> Due to the presence of three misaligned images in the SID Sony dataset (with scene IDs 10034, 10045, and 10172), our testing results in the article are based on excluding these images from the dataset. The txt file used for testing can be found and downloaded from the [Google Drive](https://drive.google.com/file/d/1nrtcZc39W4b_SJrCoMgfaO14LGem4s6O/view).


If you want to reproduce the metrics mentioned in the paper, please download the aforementioned txt file and place it in the `dataset/sid/` directory.

## Data Preparation

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Source </th>
    <th> Conf. </th>
    <th> Shot on </th>
    <th> CFA Pattern </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td> SID Sony </td>
    <th> <a href='https://cchen156.github.io/SID.html'>Learning to see in the dark</a> (<a href='https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view'>dataset only</a>) </th>
    <th> CVPR2018 </th>
    <th> Sony A7S2 </th>
    <th> Bayer (RGGB) </th>
  </tr>
  <tr>
    <td> SID Fuji </td>
    <th> <a href='https://cchen156.github.io/SID.html'>Learning to see in the dark</a> (<a href='https://drive.google.com/file/d/1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH/view'>dataset only</a>) </th>
    <th> CVPR2018 </th>
    <th> Fuji X-T2 </th>
    <th> X-Trans </th>
  </tr>
  <tr>
    <td> MCR </td>
    <th> <a href='https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark'>Abandoning the Bayer-Filter to See in the Dark</a> (<a href='https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view'>dataset only</a>) </th>
    <th> CVPR2022 </th>
    <th> MT9M001C12STC </th>
    <th> Bayer (RGGB) </th>
  </tr>
</tbody>
</table>

After download all the above datasets, you could symbol link them to the dataset folder. 
```bash
mkdir dataset && cd dataset
ln -s your/path/to/SID ./sid
ln -s your/path/to/MCR ./mcr
```
Or just put them directly in the dataset folder.

**Acceleration**

Directly training with the RAW format leads to a bottleneck on cpu. 
Thus we preprocess the SID data into numpy array for acceleration.\
Once you put the dataset in the correct place, you could simply preprocess them with the following command:
```bash
bash scripts/preprocess_sid.sh
```

Or you could preprocess with our preprocess scripts:
```bash
python scripts/preprocess/preprocess_sid.py --data-path [SID_DATA_PATH] --camera [CAM_MODEL] --split [EXP_TIME]
# [CAM_MODEL] in {Sony, Fuji}
# [EXP_TIME]  in {long, short}

# A simple example
python scripts/preprocess/preprocess_sid.py --data-path dataset/sid --camera Sony --split long
```

After all the preprocess, the final data folder should be orgnized like:
```bash
├── sid
│   ├── Fuji
│   │   ├── long
│   │   ├── long_pack
│   │   ├── long_post_int
│   │   ├── short
│   │   └── short_pack
│   ├── Sony
│   │   ├── long
│   │   ├── long_pack
│   │   ├── long_post_int
│   │   ├── short
│   │   └── short_pack
│   ├── Fuji_test_list.txt
│   ├── Fuji_train_list.txt
│   ├── Fuji_val_list.txt
│   ├── Sony_new_test_list.txt
│   ├── Sony_test_list.txt
│   ├── Sony_train_list.txt
│   └── Sony_val_list.txt
└── mcr
    ├── Mono_Colored_RAW_Paired_DATASET
    │   ├── Color_RAW_Input
    │   ├── Mono_GT
    │   └── RGB_GT
    ├── MCR_test_list.txt
    └── MCR_train_list.txt
```

## Pretrained Models

In this section, you should download the pretrained model and put them into `pretrained` folder for evalution.

<table>
<thead>
  <tr>
    <th> Trained on </th>
    <th> :link: Download Links </th>
    <th> Config file </th>
    <th> CFA Pattern </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SID Sony</td>
    <th> [<a href="https://drive.google.com/file/d/1FHreF_UHFutkiQ0LMdWjX2fahznka0Cb/view?usp=share_link">Google Drive</a>][<a href="https://pan.baidu.com/s/1-r29zUvCS-Wa2wEYovX89g?pwd=eoiz">Baidu Cloud</a>] </th>
    <th> [<a href="configs/cvpr/sony/baseline.yaml">configs/cvpr/sony/baseline</a>] </th>
    <th> Bayer (RGGB) </th>
  </tr>
  <tr>
    <td>SID Fuji</td>
    <th> [<a href="https://drive.google.com/file/d/1WfwZLBbj0EUf_QTYS8Qq5Rzk2iV8QKQ7/view?usp=share_link">Google Drive</a>][<a href="https://pan.baidu.com/s/1Sz30vAfVfF0gymNgjEUqMw?pwd=biqo">Baidu Cloud</a>]</th>
    <th> [<a href="configs/cvpr/fuji/baseline.yaml">configs/cvpr/fuji/baseline</a>] </th>
    <th> X-Trans </th>
  </tr>
  <tr>
    <td>MCR</td>
    <th> [<a href="https://drive.google.com/file/d/1kFYnqJTYfYkRWcojGxgV9DpVup4uFFBR/view?usp=share_link">Google Drive</a>][<a href="https://pan.baidu.com/s/18CjvaJZ1YtrTa_YUnQo8Vg?pwd=tkbz">Baidu Cloud</a>] </th>
    <th> [<a href="configs/cvpr/mcr/baseline.yaml">configs/cvpr/mcr/baseline</a>] </th>
    <th> Bayer (RGGB) </th>
  </tr>
</tbody>
</table>

## Evaluation

Shell scripts has been provided for benchmarking our DNF on different dataset.
```bash
bash benchmarks/[SCRIPT] [CKPT]
# [SCRIPT] in {mcr.sh, sid_sony.sh, sid_fuji.sh} determine the dataset.
# [CKPT] denotes the pretrained checkpoint.
# If you would like to save image while evaluation, you could just add `--save-image` option at the last.

# A simple example.
# To benchmark DNF on SID Sony dataset, and save the result.
bash benchmarks/sid_sony.sh pretrained/sid_sony.pth --save-image
```

## Training 

Training from scratch!
```bash
# Just use your config file!
python runner.py -cfg [CFG]
```

