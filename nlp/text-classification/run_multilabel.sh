# arbic
file=./cfg/span_emo_arabic.yml
python3 main.py -c $file
# sed -i 's/precision: 16/precision: 32/g' $file
# python3 main.py -c $file
# sed -i 's/precision: 32/precision: 16/g' $file

# english
file=./cfg/span_emo_english.yml
python3 main.py -c $file
# sed -i 's/precision: 16/precision: 32/g' $file
# python3 main.py -c $file
# sed -i 's/precision: 32/precision: 16/g' $file

# span_emo_spanish
file=./cfg/span_emo_spanish.yml
python3 main.py -c $file
# sed -i 's/precision: 16/precision: 32/g' $file
# python3 main.py -c $file
# sed -i 's/precision: 32/precision: 16/g' $file
