FPS=24
DATA_PATH=""
SAVE_PATH=""
FILE_NAME=""

while getopts ":d:s:f:" optname
do
    case "$optname" in
      d)
        DATA_PATH=$OPTARG
        ;;
      s)
        SAVE_PATH=$OPTARG
        ;;
      f)
        FILE_NAME=$OPTARG
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

echo "Generating MP4 from images..."
python scripts/generate_video_from_images.py --images-path $DATA_PATH --fps $FPS --save-path $SAVE_PATH --file-name $FILE_NAME
