models="cnn rnn rcnn cnn_bert cnn_roberta cnn_deberta"
# models="cnn_roberta cnn_deberta"
for model in $models
do
    file="./cfg/text_$model.yml"
    python3 main.py -c $file
    sed -i 's/precision: 16/precision: 32/g' $file
    python3 main.py -c $file
    sed -i 's/precision: 32/precision: 16/g' $file
done
