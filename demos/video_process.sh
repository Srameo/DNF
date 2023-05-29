export PYTHONPATH=$PWD
FPS=24
PRETRAIN_CKPT=""
DATA_PATH=""
SAVE_PATH=""
FILE_NAME=""
RATIO="50"

while getopts ":p:d:s:f:r:" optname
do
    case "$optname" in
      p)
        PRETRAIN_CKPT=$OPTARG
        ;;
      d)
        DATA_PATH=$OPTARG
        ;;
      s)
        SAVE_PATH=$OPTARG
        ;;
      f)
        FILE_NAME=$OPTARG
        ;;
      r)
        RATIO=$OPTARG
        ;;
      :)
        echo "No argument value for option $OPTARG"
        ;;
      ?)
        echo "Unknown option $OPTARG"
        ;;
      *)
        echo "Unknown error while processing options"
        ;;
    esac
    #echo "option index is $OPTIND"
done

python scripts/inference.py -cfg configs/demo/shell_base.yaml \
  --pretrain $PRETRAIN_CKPT \
  --force-yml data:data_path=$DATA_PATH \
              data:test:ratio=$RATIO
echo "Generating MP4 from images..."
python scripts/generate_video_from_images.py \
  --images-path runs/CVPR_DEMO/video_demo/results/inference \
  --fps $FPS \
  --save-path $SAVE_PATH \
  --file-name $FILE_NAME
