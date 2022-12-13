export CUDA_VISIBLE_DEVICES=1 

# english
file=./cfg/span_emo_english.yml
python3 main.py -c $file -s 23
python3 main.py -c $file -s 312
python3 main.py -c $file -s 5123
python3 main.py -c $file -s 7123
python3 main.py -c $file -s 11123
sed -i 's/precision: 16/precision: 32/g' $file
python3 main.py -c $file -s 23
python3 main.py -c $file -s 312
python3 main.py -c $file -s 5123
python3 main.py -c $file -s 7123
python3 main.py -c $file -s 11123
sed -i 's/precision: 32/precision: 16/g' $file

# span_emo_spanish
file=./cfg/span_emo_spanish.yml
python3 main.py -c $file -s 2222
python3 main.py -c $file -s 3333
python3 main.py -c $file -s 5455
python3 main.py -c $file -s 7677
python3 main.py -c $file -s 11123
sed -i 's/precision: 16/precision: 32/g' $file
python3 main.py -c $file -s 2222
python3 main.py -c $file -s 3333
python3 main.py -c $file -s 5455
python3 main.py -c $file -s 7677
python3 main.py -c $file -s 11123
sed -i 's/precision: 32/precision: 16/g' $file

# arbic
file=./cfg/span_emo_arabic.yml
python3 main.py -c $file -s 2
python3 main.py -c $file -s 3
python3 main.py -c $file -s 5
python3 main.py -c $file -s 7
python3 main.py -c $file -s 11
sed -i 's/precision: 16/precision: 32/g' $file
python3 main.py -c $file -s 2
python3 main.py -c $file -s 3
python3 main.py -c $file -s 5
python3 main.py -c $file -s 7
python3 main.py -c $file -s 11
sed -i 's/precision: 32/precision: 16/g' $file
