# Try DNF!

- [Try DNF!](#try-dnf)
  - [Provided Data](#provided-data)
  - [Convert Your Own RAW Images to Numpy for Acceleration](#convert-your-own-raw-images-to-numpy-for-acceleration)
  - [Convert Your Own RAW Video](#convert-your-own-raw-video)
  - [Tips for Scripts](#tips-for-scripts)

## Provided Data

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Download Links </th>
    <th> Shot on </th>
    <th> CFA Pattern </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>NKU Campus</td>
    <th> [<a href="https://drive.google.com/drive/folders/1XNgQSebMsLRxxhNDP6xV4C5zU5lwiALA?usp=share_link" >Google Drive</a>][<a href="https://pan.baidu.com/s/16HEq0f4_E7gYiw581Tpp7A?pwd=30wa">Baidu Cloud</a>] </th>
    <th> Synthetic </th>
    <th> Bayer (RGGB) </th>
  </tr>
</tbody>
</table>

After Downloading the final data folder should be organized like:
```bash
Campus
├── list.txt
├── long_pack      # Optional, if the dataset was synthetic
│   └── *.npy
├── long_post_int  # Optional, if the dataset was synthetic
│   └── *.npy
├── long_png       # Optional, if the dataset was synthetic
│   └── *.png
├── short_pack
│   └── *.npy
└── short_png
    └── *.png
```

## Convert Your Own RAW Images to Numpy for Acceleration

If your raw data is with Sony `ARW` format, you could simply convert the data folder `[DIR]` by the following command:
> **Notice**: your data should be saved in `[DIR]/short` folder.
```bash
python scripts/preprocess/preprocess_sid.py --data-path [DIR] --camera Sony --split short
```
Then the numpy array format of your own raw image could be found in `[DIR]/short_pack`.

## Convert Your Own RAW Video

TBD

## Tips for Scripts

All the shell scripts for demo could be found in `demo/` folder.\
Besides the `images_process.sh` and `video_process.sh` as described in [README](../README.md), the `images_to_video.sh` could transform a image sequenes into a video clip:
```bash
bash demos/images_to_video.sh -d [DIR] -s [SAVE_PATH] -f [FILE_NAME]
# [DIR] is the path to your images (jpg or png format).
# Your video clip could be found in [SAVE_PATH]/[FILE_NAME].mp4

# A simple example
bash demos/images_to_video.sh\
 -d dataset/Campus \
 -s runs/video \
 -f campus_short
```
