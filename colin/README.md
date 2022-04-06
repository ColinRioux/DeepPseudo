# DeepPseudo Extension
Author: Colin Rioux

```bash
conda create -n coen497
conda activate coen497
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install nltk numpy scikit-learn scikit-image matplotlib torchtext==0.9.0
pip install tokenizers boto3 filelock requests tqdm regex sentencepiece sacremoses
pip install git+https://github.com/Maluuba/nlg-eval.git@master
pip install pandas pytextrank
nlg-eval --setup ./DeepPseudo/data/django
nlg-eval --setup ./DeepPseudo/data/spoc
python -m spacy download en_core_web_sm
```
