export PYTHONPATH=$PWD
PRETRAIN_CKPT=""
DATA_PATH=""
RATIO="100"
ACC="false"

while getopts ":p:d:r:a" optname
do
    case "$optname" in
      p)
        PRETRAIN_CKPT=$OPTARG
        ;;
      d)
        DATA_PATH=$OPTARG
        ;;
      r)
        RATIO=$OPTARG
        ;;
      a)
        ACC="true"
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

python scripts/inference.py -cfg configs/inference/shell_base.yaml --pretrain $PRETRAIN_CKPT --force-yml data:data_path=$DATA_PATH data:test:ratio=$RATIO data:load_npy=$ACC
