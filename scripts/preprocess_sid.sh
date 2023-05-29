echo 'Preprocess SID Sony long...'
python scripts/preprocess/preprocess_sid.py --data-path dataset/sid --camera Sony --split long 

echo 'Preprocess SID Sony short...'
python scripts/preprocess/preprocess_sid.py --data-path dataset/sid --camera Sony --split short 

echo 'Preprocess SID Fuji long...'
python scripts/preprocess/preprocess_sid.py --data-path dataset/sid --camera Fuji --split long 

echo 'Preprocess SID Fuji long...'
python scripts/preprocess/preprocess_sid.py --data-path dataset/sid --camera Fuji --split short 
