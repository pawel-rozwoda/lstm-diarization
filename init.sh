echo "DATA_PATH='$(pwd)/data/'" > ./src/data_path.py

mkdir data/callhome/
cd data/callhome/
rm -r eng/
wget https://ca.talkbank.org/data/CallHome/eng.zip
unzip eng.zip
rm eng.zip

wget -r -A .wav https://media.talkbank.org/ca/CallHome/eng/0wav/4289.wav
mv media.talkbank.org/ca/CallHome/eng/0wav/ wav/
rm -r media.talkbank.org

