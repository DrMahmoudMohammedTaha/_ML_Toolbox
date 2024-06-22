
# to get all current versions
pip freeze > requirements.txt 

# install local library file
python -m pip install dlib-19.22.99-cp310-cp310-win_amd64.whl 

# get the path of any installed lib
where conda

# additional
pip install cmake
conda install -c conda-forge dlib

# run uvicorn app
uvicorn main:app --reload

# copy file
!cp "/content/drive/My Drive/TAI-sorted.zip"  "/content/json"

## unzip metadate file
!tar -xvf  '/content/drive/My Drive/dataSet/danbooru2019/metadata.json.tar.xz' -C '/content/drive/My Drive/dataSet/danbooru2019/metadata'
!unzip "/content/drive/My Drive/TAI-important.zip" -d "/content"

# compress folder
!zip -r "e_images.zip" "/content/e_img" 

# kaggle
!pip install -q kaggle
!pip install -q kaggle-cli
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets list -s tagged-anime-illustrations
!kaggle datasets download -d mylesoneill/tagged-anime-illustrations

## download images folder in original folder
for x in range(99,100):
  !rsync --recursive --verbose rsync://78.46.86.149:873/danbooru2019/original/00{x} "/content/drive/My Drive/dataSet/danbooru2019/original"
  


