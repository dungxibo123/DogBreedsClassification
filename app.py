import streamlit as st
from contextlib import contextmanager
from io import StringIO

from threading import current_thread
from datetime import datetime
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from mock import *
import sys
import torch
import os
from model import *
import matplotlib.pyplot as plt
st.set_page_config(page_title="Dog Breeds Classification",layout='wide')
# Create two columns
col1, col2, col3 = st.columns(3)

# Add dropdown options to the left column
with col1:
    st.header("Configuration")
    pretrain_model = st.selectbox("Select a pretrained model", ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"], key="model")
    checkpoint_pth = st.text_input("Checkpoint save path (./....)", './checkpoint/',key="checkpoint_pth",placeholder="./checkpoint/")
    data_folder = st.text_input("Data Folder", './data/images/CloneImages', key="data_folder", placeholder="./data/images/CloneImages")
    resume_from_checkpoint = st.text_input("Resume from checkpoint", "None", key="resume_from_checkpoint")
    checkpoint_save_freq = st.text_input("Checkpoint Saving Frequency", "5", key="checkpoint_save_freq")
    use_gpu_list = ["Yes", "No"] if torch.cuda.is_available() else ["No"]
    use_gpu = st.selectbox("Use GPU", use_gpu_list, key="use_gpu")

with col2:
    st.header("Training option")
    num_classes = len(os.listdir(data_folder))
    optimizer = st.selectbox("Optimizer" , ["sgd", "adam", "adamw"], key="optimizer")
    batch_size = st.text_input("Batch size", "8", key="batch_size")
    val_batch_size = st.text_input("Validation batch size", "32", key="val_batch_size")
    epoch = st.text_input("Number of Epoch", "400", key="epoch")
    learning_rate = st.text_input("Learning Rate", "0.005", key='learning_rate')

    weight_decay = st.text_input("Weight Decay", "0.004", key='weight_decay')


with col3:
    st.header("...")
    num_hidden = st.text_input("Number of perceptrons before last layer", "128", key="num_hidden")
    validation_split = st.text_input("Validation Split", "0.2", key="validation_split")
    dropout = st.text_input("Dropout rate", "0.35", key="dropout")
    patience = st.text_input("Early stopping patience", "5", key="patience")



    











@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write



def click_train_button():
    with st_stdout("info"):
        opt = {
            "data_folder": data_folder,
            "checkpoint_pth": checkpoint_pth,
            "num_classes": num_classes,
            "batch_size": int(batch_size),
            "val_batch_size": int(val_batch_size),
            "use_gpu": True if use_gpu == "Yes" else False,
            "resume_from_checkpoint": False if resume_from_checkpoint is None or resume_from_checkpoint == "None" else True,
            "epoch": int(epoch),
            "early_stopping": True,
            "patience": int(patience),
            "model": pretrain_model,
            "pretrained_weights": "IMAGENET1K_V1",
            "optimizer": optimizer,
            "learning_rate": float(learning_rate),
            "validation_split": float(validation_split),
            "dropout": float(dropout),
            "checkpoint_pth": './checkpoint',
            "continual_checkpoint_pth": resume_from_checkpoint,
            "num_hidden": int(num_hidden),
            "checkpoint_save_freq": int(checkpoint_save_freq),
            "weight_decay": float(weight_decay),
            "save_checkpoint": True,
        }
        #print(opt)
        model, losses, val_losses, train_acc, val_acc = training_process(opt)
    fig = plt.figure(figsize=(10,10))
    fig.gca().plot([i for i in range(1, len(losses) + 1)], losses, label="Training loss")
    fig.gca().plot([i for i in range(1, len(losses) + 1)], val_losses, label = "Validation Loss")
    fig1 = plt.figure(figsize=(10,10))
    fig1.gca().plot([i for i in range(1, len(losses) + 1)], train_acc, label="Training accuracy")
    fig1.gca().plot([i for i in range(1, len(losses) + 1)], val_acc, label = "Validation accuracy")

    return fig, fig1

            

train_button = st.button("Start training!", key="train_button")


if train_button:
    fig, fig1 = click_train_button()
    st.pyplot(fig)
    st.pyplot(fig1)
