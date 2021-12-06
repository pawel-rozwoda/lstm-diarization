echo "DATA_PATH='$(pwd)/data/'" > ./src/data_path.py

mkdir -p data/callhome/
cd data/callhome/
wget https://ca.talkbank.org/data/CallHome/eng.zip
unzip eng.zip
rm eng.zip

wget --no-parent -r -A .wav https://media.talkbank.org/ca/CallHome/eng/0wav/
mv media.talkbank.org/ca/CallHome/eng/0wav/ wav/
rm -r media.talkbank.org

