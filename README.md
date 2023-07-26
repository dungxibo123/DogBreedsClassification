# Dogs Breed Classification

## 1. Installation

Make sure you can use [Kaggle API](https://www.kaggle.com/docs/api).

```bash
> git clone https://github.com/dungxibo123/DogBreedsClassification
> cd DogBreedsClassification
> pip install -r requirements.txt
> pip install -U --force-reinstall protobuf==3.13.0 altair==4.0.0 # Due to the confliction some where inside the streamlit package
```

**Note:** If you have a GPU device, make sure you have installed the PyTorch package with compatible cuda version.
## 2. Preparation

```bash
> pip install -U kaggle
> kaggle datasets download -d jessicali9530/stanford-dogs-dataset
> unzip -qq stanford-dogs-dataset.zip -d data/


# Preparing mock data (use can define the number of class that you want to copy to the mock folder at line 16 [0:<your desire number of classes]
> python setup.py
```

## 3. Running the streamlit
Simply run
```bash
> streamlit run app.py
```



Everything should be fine now. If it is not, contact me :'(



