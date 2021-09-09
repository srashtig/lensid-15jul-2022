import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import add


# import plotsettings
import tensorflow as tf
import matplotlib.image as mpimg
from skimage.io import imread
import os

import cv2
import xgboost as xgb
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    concatenate,
    LeakyReLU,
    Input,
    Dense,
    Flatten,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
    plot_roc_curve,
)
from sklearn.impute import SimpleImputer
import joblib
import time
import datetime
from sklearn.model_selection import StratifiedKFold


def lrfn(epoch):
    """Helper function for training densnets, returns the learning rate at 
    each epoch."""
    LR_START = 0.00001
    LR_MAX = 0.00005  # * strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = 0.8

    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return lr


def train_densenet(input_matrix, labels, det, epochs, kernel_lr=1):
    """
    Train the modified Densenet CNN , given the input feature matrix
        (generated from Qtransforms) and labels.

    Parameters:
        input_matrix (4-dim numpy array(n,128,128,3)): superimposed 
            RGB images of Qtransformsfor n event pairs.
            
        labels(1-d numpy array (n,1)): array of lensed(0) and ones(1).
        
        det(str): either of the three detectors 'H1','L1','V1'.
        
        epochs(int): no. of gradient descent steps to take.
        
        kernel_lr(float): l2 regulariser parameter, default: 1
        
    Returns:
        model(Keras model): trained on the input matrix and labels.
    """
    
    
    '''
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=40*1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    '''
    
    pre_model = tf.keras.applications.DenseNet201(
        input_shape=(128, 128, 3), weights="imagenet", include_top=False
    )

    pre_model.trainable = True
    model = tf.keras.Sequential(
        [
            pre_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(kernel_lr),
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.summary()
    f = pre_model.name.replace(".", "_") + "_det_" + det + ".h5"
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
    mc_trained = ModelCheckpoint(
        "trained_" + f,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    mc_untrained = ModelCheckpoint(
        "untrained_" + f,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    start_time = time.time()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    if pre_model.trainable == True:
        history = model.fit(
            input_matrix[:, :, :, :3],
            labels,
            batch_size=32,
            epochs=epochs,
            callbacks=[lr_schedule, mc_trained, es],
            verbose=1,
            validation_split=0.2,
            shuffle=True,
        )

    elif pre_model.trainable == False:
        history = model.fit(
            input_matrix[:, :, :, :3],
            labels,
            batch_size=32,
            epochs=epochs,
            callbacks=[lr_schedule, mc_untrained, es],
            verbose=1,
            validation_split=0.2,
            shuffle=True,
        )
    end_time = time.time()
    print("Time Taken: ", end_time - start_time)

    return model


class generate_resize_densenet_fm:
    """
    Class to generate input feature matrix to the DenseNets using Qtransform
        images.

    Attributes:
        df(pandas dataframe): dataframe with img_0, img_1 and Lensing as the 
            columns for loading the Qtransform images using img_0 and img_1
            values and train using 'Lensing' as labels.
    """

    def __init__(self, df):
        """
        Constructor to generate densenet input feature matrix class.

        Parameters:
            df(pandas dataframe): dataframe with img_0, img_1 and Lensing as
                the columns for loading the Qtransform images using img_0 and 
                img_1 values and train using 'Lensing' as labels.
        """
        self.df = df

    def read_spectograms(self, file_paths, img_rows, img_cols, channels):
        """
        Reads and resizes Qtransform images from the storage given the file paths.

        Parameters:
            file_paths(list): List containing the paths of the Qtransform 
                images to be read.
                
            img_rows(int): width of the desired Qtransform images.
            
            img_cols(int): height of the desired Qtransform images.
            
            channels(int): Depth of each Qtransform image(RGB=3).

        Returns:
            4-dim numpy array(n,img_rows,img_cols,channels): stacked images
                of the n Qtransforms of size (img_rows,img_cols,channels).
                
            list(missing_ids): list of ids for which QT images are not found
                in the given file paths.
        """
        images = []
        missing_ids = []
        for i, file_path in enumerate(file_paths):
            if os.path.isfile(file_path) == True:
                img = cv2.imread(file_path)
                img = cv2.resize(img, (img_rows, img_cols))
            else:
                print(' %s not found'%file_path)
                img = np.zeros([img_rows, img_cols, channels])
                missing_ids.append(i)
            images.append(img)

        images = np.asarray(images, dtype=np.float32)

        # normalize
        images = images / np.max(images)

        # reshape to match Keras expectations
        images = images.reshape(images.shape[0], img_rows, img_cols, channels)

        return images, missing_ids

    def DenseNet_input_matrix(
        self,
        det,
        data_mode_dense="current",
        data_dir="../../data/qts/test/",
        phenom=False,
        whitened=False,
    ):
        """
        Reads and resizes Qtransform images from the storage given the file
            paths.

        Parameters:
            det(str): either of the three detectors 'H1','L1','V1'.
            
            data_dir(str): path of the directory containing the Qtransform 
                images.
                
            phenom(bool): to addtionally compute hand derived features form
                the Qtransforms and add to the dataframe. default: False.
                
            whitened(bool): to use the whitened Qtransformed images. 
                default: False.

        Returns:
            4-dim numpy array(n,128,128,3): superimposed RGB images of 
                Qtransforms for n event pairs.
                
            list(size n): labels of event pairs (Lensing column of the 
                input dataframe).
                
            list: missing_ids, containing the row ids of the dataframe 
                for which Qtransform images are not found.
                
            pandas dataframe(returned only if phenom=True): adds the 
                hand derived features using the Qtransform images as 
                columns to the input dataframe.
        """
        df = self.df
        in_channel = 3

        # height and width
        img_rows, img_cols = 128, 128
        input_shape = (img_rows, img_cols, in_channel)
        input_img = Input(shape=input_shape)

        prefix_paths = det + "/"
        suffix_paths = ".png"
        if whitened == True:
            suffix_paths = "-whitened.png"

        img_0_paths = add(
            add(data_dir, add(prefix_paths, df.img_0.values.astype(str))),
            suffix_paths,
        )
        img_1_paths = add(
            add(data_dir, add(prefix_paths, df.img_1.values.astype(str))),
            suffix_paths,
        )

        x_0, missing_ids_0 = self.read_spectograms(
            img_0_paths, img_rows, img_cols, in_channel
        )
        x_1, missing_ids_1 = self.read_spectograms(
            img_1_paths, img_rows, img_cols, in_channel
        )
        labels = df.Lensing.values
        ll = len(labels)
        x_comp1 = np.zeros([ll, img_rows, img_cols, in_channel])
        mean_overlap_qts, std_overlap_qts, lsq_qts = (
            np.zeros(ll),
            np.zeros(ll),
            np.zeros(ll),
        )

        for idx in range(ll):
            x_comp1[idx, :, :, :] = (x_0[idx, :, :, :] + x_1[idx, :, :, :]) / 2
            mean_overlap_qts[idx] = (
                x_0[idx, :, :, :] * x_1[idx, :, :, :]
            ).sum(axis=(0, 1, 2))
            std_overlap_qts[idx] = (x_0[idx, :, :, :] * x_1[idx, :, :, :]).std(
                axis=(0, 1, 2)
            )
        lsq_qts = np.abs(x_0 - x_1).sum(axis=(1, 2, 3))
        (
            df["mean_overlap_qts_" + det],
            df["std_overlap_qts_" + det],
            df["lsq_overlap_qts_" + det],
        ) = (mean_overlap_qts, std_overlap_qts, lsq_qts)
        missing_ids = np.union1d(missing_ids_0, missing_ids_1).astype(int)
        if phenom == False:
            return x_comp1, labels, missing_ids
        else:
            return x_comp1, labels, missing_ids, df


def Dense_predict(model, df, input_matrix, missing_ids):
    """
    Outputs predictions from the trained Densenet model using the given input
    feature matrix formed by the superimosed Qtransform images.

    Parameters:
        model(Keras model): trained densenet model.
        
        df(pandas Dataframe): input data frame.
        
        input_matrix(4-dim numpy array(n,128,128,3): superimposed RGB images
            of Qtransforms for n event pairs.
            
        missing_ids(list): containing the row ids of the dataframe for which
            Qtransform images are not found.

    Returns:
        1d numpy array: predictions of the Densenet model[0,1].

    """
    y_predict = model.predict(input_matrix[:, :, :, :3])
    y_predict[missing_ids] = None
    return y_predict


def train_xgboost_dense_qts(
    df_train,
    from_df=True,
    model_id_dense=0,
    n_estimators=135,
    max_depth=6,
    scale_pos_weight=0.01,
    include_phenom=False,
):
    """
    Train XGBoost Algorithm for the Qtransforms, which takes input as the 
    three detector predictions of the trained DenseNet models, and optionally
    the hand derieved features from Qtransforms, for given set of event pairs.

    Parameters:
        df_train(pandas Dataframe): Dataframe containing the img_0, img_1, 
            Lensing(labels), and the three Densenet predictions for the
            three detectors, eg: 'dense_H1_0' etc.. Optionally should have
            hand derieved features as columns.
            
        model_id_dense(int): Identifier for the DenseNet prediction columns
            in the Dataframe. Default: 0.
            
        n_estimators(int): hyperparameter of XGBoost, setting the maximum 
            no. of trees. Default: 135.
            
        max_depth(int): hyperparameter of XGBoost, setting the maximum size 
            of trees. Default: 6.
            
        scale_pos_weight(float): hyperparameter of XGBoost, setting the weight
            of the unbalanced dataset, eg. ratio of lensed to unlensed event 
                rates. Default: 0.01.
                
        include_phenom(bool): include hand dereived features also while 
            training. Default: False.

    Returns:
        the trained XGBoost model
    """
    cols = [
        "dense_H1_" + str(model_id_dense),
        "dense_L1_" + str(model_id_dense),
        "dense_V1_" + str(model_id_dense),
    ]
    if include_phenom == True:
        phenom_features = [
            "mean_overlap_qts_H1",
            "std_overlap_qts_H1",
            "lsq_overlap_qts_L1",
            "mean_overlap_qts_L1",
            "std_overlap_qts_L1",
            "lsq_overlap_qts_V1",
            "mean_overlap_qts_V1",
            "std_overlap_qts_V1",
            "lsq_overlap_qts_V1",
        ]
        cols = cols + phenom_features
    if from_df == True:
        xgb_qts_train, labels = np.c_[df_train[cols]], df_train.Lensing.values
    else:
        print("Dense predictions not in df")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X=xgb_qts_train, y=labels)
    return model


def predict_xgboost_dense_qts(
    df,
    model,
    model_id_dense=0,
    model_id_xgb=0,
    from_df=True,
    fill_missing=True,
    include_phenom=False,
):
    """
    Adds XGBoost Algorithm predictions for the Qtransforms, which takes input
    as the three detector predictions of the trained DenseNet models, and
    optionally the hand derieved features from Qtransforms, for given set of 
    event pairs.

    Parameters:
        df(pandas Dataframe): Dataframe containing the img_0, img_1, 
            Lensing(labels), and the three Densenet Predictions for the three
            detectors, eg: 'dense_H1_0' etc.. Optionally should have hand
            derieved features as columns.
            
        model_id_dense(int): Identifier for the DenseNet prediction columns 
            in the Dataframe. Default: 0.
            
        model_id_xgb(int): Identifier for adding the XGBoost prediction column
            in the Dataframe. Default: 0.
            
        fill_missing(bool): In case of missing Qtransform images, or DenseNet 
            predictions, return the product of the predictions for the 
            available detector DenseNet predicitons, other return None.
            Default: True
            
        include_phenom(bool): include hand dereived features also while 
            training. Default: False.

    Returns:
        pandas Dataframe: the input dataframe with additional column eg.
        'xgb_dense_QTS_0' , as the given XGBoost with Qtransforms predictions
        for the event pairs.
    """
    cols = [
        "dense_H1_" + str(model_id_dense),
        "dense_L1_" + str(model_id_dense),
        "dense_V1_" + str(model_id_dense),
    ]
    if include_phenom == True:
        phenom_features = [
            "mean_overlap_qts_H1",
            "std_overlap_qts_H1",
            "lsq_overlap_qts_L1",
            "mean_overlap_qts_L1",
            "std_overlap_qts_L1",
            "lsq_overlap_qts_V1",
            "mean_overlap_qts_V1",
            "std_overlap_qts_V1",
            "lsq_overlap_qts_V1",
        ]
        cols = cols + phenom_features

    if from_df == True:
        x1, x2, x3 = df[cols[0]], df[cols[1]], df[cols[2]]
        missing_out = (x1.isna() | x2.isna() | x3.isna()).values
        missing_all = (x1.isna() & x2.isna() & x3.isna()).values

    else:
        print("Dense predictions not in df")
    input_matrix = np.c_[df[cols]]

    y_predict = model.predict_proba(input_matrix)[:, 1]
    if fill_missing == True:
        y_predict[missing_out] = (
            x1[missing_out].fillna(1).values
            * x2[missing_out].fillna(1).values
            * x3[missing_out].fillna(1).values
        )
        y_predict[missing_all] = None

    else:
        y_predict[missing_out] = None
    df["xgb_dense_QTS_" + str(model_id_xgb)] = y_predict

    return df


class generate_skymaps_fm:
    """
    Class to generate input feature matrix to the XGBoost using skymaps.

    Attributes:
        df(pandas dataframe): dataframe with img_0, img_1 and Lensing as the 
        columns for loading the Qtransform images using img_0 and img_1 values
        and train using 'Lensing' as labels
    """

    def __init__(self, df):
        """
        Constructor to generate XGBoost with skymaps input feature matrix class.

        Parameters:
            df(pandas dataframe): dataframe with 'img_0', 'img_1' as the 
                columns for loading the skymaps files(.npz) using img_0 
                and img_1 values and the 'Lensing' columns as labels.
        """
        self.df = df

    def load_data(self, img_paths, THETA_g):
        """
        Read the .npz skymaps and normalise them.

        Parameters:
            img_paths(list): list of 'n' paths of the .npz files for the
                skymaps.
                
            THETA_g(2d numpy array of size 400x800): The grid of 'theta' 
                (declination) over the 2d sky.

        Returns:
            3d numpy array(n,400,800): stacked, cartesian and normalised
                skymaps for the given paths.
        """
        combine = []
        missing_ids = []
        for i,img in enumerate(img_paths):
            try:
                data = np.load(img)["data"]
            except: 
                print('%s not found'%img)
                data=np.zeros([400,800])
                missing_ids.append(i)
            data /= (
                data * np.sin(THETA_g).T * 2 * np.pi * np.pi / (400 * 800)
            ).sum()
            combine.append(data)
        combine = np.asarray(combine)
        return combine,missing_ids

    def XGBoost_input_matrix(
        self,
        data_mode_xgb="bayestar_skymaps",
        data_dir="../../data/bayestar_skymaps/test/",
    ):
        """
        Compute, return and add the features for skymaps for the given event 
        pairs from the .npz files.

        Parameters:
            data_mode_xgb('bayestar_skymaps' or 'pe_skymaps'): to load and 
                construct features for bayestar skymaps(defaut) or PE skymaps.
                
            data_dir(str): path of the directory that contains the skymaps 
                datafiles(.npz).

        Returns:
            2d numpy array(n,4): array with the four features for each pair
                of events.
                
            labels(list): list containing the labels ('Lensing' column of 
                Dataframe) for the given pairs.
                
            pandas Dataframe:  Input dataframe with additional four columns as
                the features constructed from the skymaps.
        """
        df = self.df
        scaler = StandardScaler()
        my_imputer = SimpleImputer()
        img_row, img_col = 400, 800
        image_vector_size = img_row * img_col
        THETA, PHI = np.linspace(0, np.pi, img_row), np.linspace(
            0, 2 * np.pi, img_col
        )
        THETA_g, PHI_g = np.meshgrid(THETA, PHI)

        img_0_paths = add(add(data_dir, df.img_0.values.astype(str)), ".npz")
        img_1_paths = add(add(data_dir, df.img_1.values.astype(str)), ".npz")

        img_0,missing_ids_0 = self.load_data(img_0_paths, THETA_g)
        img_1,missing_ids_1 = self.load_data(img_1_paths, THETA_g)
        blu, d2, lsq, d3 = self.calc_features(img_0, img_1)
        missing_ids=np.union1d(missing_ids_0,missing_ids_1)
        labels = df.Lensing.values

        del img_0, img_1

        Input_combined = np.array([blu, d2, lsq, d3]).T
        Input_combined = my_imputer.fit_transform(Input_combined)

        if (data_mode_xgb == "pe_skymaps") or (data_mode_xgb == "pe"):
            df = df.assign(
                pe_skymaps_blu=blu,
                pe_skymaps_d2=d2,
                pe_skymaps_lsq=lsq,
                pe_skymaps_d3=d3,
            )
        else:
            df = df.assign(
                bayestar_skymaps_blu=blu,
                bayestar_skymaps_d2=d2,
                bayestar_skymaps_lsq=lsq,
                bayestar_skymaps_d3=d3,
            )

        return Input_combined, labels, df, missing_ids

    def calc_features(self, img_0, img_1):
        """Returns the four features for the given pair(img_0 and img_1) of 
            cartesian skymaps.

        Parameters:
            img_0(numpy array(400,800)): normalised skymap of first event in 
                the pair.
                
            img_1(numpy array(400,800)): normalised skymap of second event in 
                the pair.

        Returns:
            float: k1 = sum(img_0*img_1)
            
            float: d2 = mean(img_0*img_1)
            
            float: k2 = mean(abs(img_0-img_1))
            
            float: k3 = std(img_0*img1)

        """
        prior_ra = 1 / (2 * np.pi)
        blu = (img_0 * img_1).sum(axis=(1, 2)) * (np.pi / 400) ** 2 / prior_ra
        d2 = np.mean(img_0 * img_1, axis=(1, 2))
        lsq = np.mean(np.abs(img_0 - img_1), axis=(1, 2))
        d3 = np.std(img_0 * img_1, axis=(1, 2))
        return blu, d2, lsq, d3

    def XGBoost_input_matrix_from_df(self, data_mode_xgb="bayestar_skymaps"):
        """
        Returns the standardised features and labels for XGBoost sky from the 
            dataframe itself.

        Parameters:
            data_mode_xgb('bayestar_skymaps' or 'pe_skymaps'): to load and 
                construct features for bayestar skymaps(defaut) or PE skymaps.

        Returns:
            2d numpy array(n,4): array with the four features for each pair of 
                events.
                
            labels(list): list containing the labels ('Lensing' column of 
                Dataframe) for the given pairs.
                
            pandas Dataframe:  Input dataframe.
        """

        df = self.df
        scaler = StandardScaler()
        my_imputer = SimpleImputer()
        if data_mode_xgb == "pe_skymaps" or (data_mode_xgb == "pe"):
            blu = df["pe_skymaps_blu"]
            d2 = df["pe_skymaps_d2"]
            lsq = df["pe_skymaps_lsq"]
            d3 = df["pe_skymaps_d3"]
        else:
            blu = df["bayestar_skymaps_blu"]
            d2 = df["bayestar_skymaps_d2"]
            lsq = df["bayestar_skymaps_lsq"]
            d3 = df["bayestar_skymaps_d3"]

        labels = df.Lensing.values
        Input_combined = np.array([blu, d2, lsq, d3]).T
        Input_combined = my_imputer.fit_transform(Input_combined)
        return Input_combined, labels, df


def XGB_predict(df, model, data_mode_xgb="bayestar_skymaps"):
    """
    Adds XGBoost Algorithm predictions for the skymaps given the trained
    XGBoost model and sky features in the dataframe.

    Parameters:
        df(pandas Dataframe): Dataframe containing the img_0, img_1, 
            Lensing(labels), and the sky features as columns, 
            'bayestar_skymaps_blu', 'bayestar_skymaps_d2',
            'bayestar_skymaps_d3', 'bayestar_skymaps_lsq'.
            
        data_mode_xgb('bayestar_skymaps' or 'pe_skymaps'): to predict using 
            the features for bayestar skymaps(defaut) or PE skymaps.

    Returns:
        pandas Dataframe: the input dataframe with additional column eg. 
        'xgb_pred_bayestar_skymaps' , as the given XGBoost with Skymaps 
        predictions for the event pairs.
    """

    if data_mode_xgb == "pe_skymaps" or (data_mode_xgb == "pe"):
        blu = df["pe_skymaps_blu"]
        d2 = df["pe_skymaps_d2"]
        lsq = df["pe_skymaps_lsq"]
        d3 = df["pe_skymaps_d3"]
    else:
        blu = df["bayestar_skymaps_blu"]
        d2 = df["bayestar_skymaps_d2"]
        lsq = df["bayestar_skymaps_lsq"]
        d3 = df["bayestar_skymaps_d3"]
    x_sky = np.c_[blu, d2, lsq, d3]
    y_predict = model.predict_proba(x_sky)[:, 1]
    if data_mode_xgb == "pe_skymaps" or (data_mode_xgb == "pe"):
        df_xgb = df.assign(xgb_pred_pe_skymaps=y_predict)
    else:
        df_xgb = df.assign(xgb_pred_bayestar_skymaps=y_predict)

    return df_xgb


def train_xgboost_sky(
    df_train, n_estimators=110, max_depth=6, scale_pos_weight=0.01
):
    """
    Train XGBoost Algorithm for the Skymaps, which takes input as the sky 
    features from the input dataframe or .npz files, for given set of event
    pairs.

    Parameters:
        df_train(pandas Dataframe): Dataframe containing the img_0, img_1,
            Lensing(labels), and the sky features as columns, 
            'bayestar_skymaps_blu', 'bayestar_skymaps_d2',
            'bayestar_skymaps_d3', 'bayestar_skymaps_lsq'.
            
        n_estimators(int): hyperparameter of XGBoost, setting the maximum 
            no. of trees. Default: 110.
            
        max_depth(int): hyperparameter of XGBoost, setting the maximum size
            of trees. Default: 6.
            
        scale_pos_weight(float): hyperparameter of XGBoost, setting the weight 
            of the unbalanced dataset, eg. ratio of lensed to unlensed event
            rates. Default: 0.01.

    Returns:
        the trained XGBoost with skymaps model
    """
    xgb_sky_train, xgb_sky_label_train, df_train = generate_skymaps_fm(
        df_train
    ).XGBoost_input_matrix_from_df(data_mode_xgb="bayestar_skymaps")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X=xgb_sky_train, y=xgb_sky_label_train)
    del xgb_sky_train, xgb_sky_label_train
    return model


def plot_ROCs(
    df,
    logy=False,
    cols=[
        "dense_H1",
        "dense_L1",
        "dense_V1",
        "xgb_pred_bayestar_skymaps",
        "combined_pred_bayestar_skymaps",
    ],
    labels=None,
    ylim=0,
):
    """
    Plots the ROC(reciever operating curves, Efficiency v/s False Positive 
    Probability) from the input dataframe, for the given set of columns.

    Parameters:
        df(pandas Dataframe): dataframe with the event pairs('img_0','img_1')
            and the ranking statistics as columns, along with the labels in a
            column ('Lensing').
            
        logy(bool): plot the yscale in log, True/False(default).
        
        cols(list): list of the column names for which ROCs should be plot.
        
        labels(list): list of the labels to be put as legend in the plots for 
            the corresponding ranking statistic columns. If not given the cols
            will be used in legend.
            
        ylim(float): The lower limit of the y-axis of the plot.

    Returns:
        Matplotlib figure:  with the plots of ROCs
        dict: each ranking statistic's FPP, efficiency and thresholds.

    """
    if labels == None:
        labels = cols
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    rocs = {}
    for i, col in enumerate(cols):
        y_test = df.Lensing.values
        fig = plt.figure(1, figsize=(10, 7))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_test, df[col]
        )
        rocs[col] = (false_positive_rate, true_positive_rate, thresholds)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            "-",
            label=labels[i],
            color=colors[i],
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print(labels[i], "auc = %.4f" % roc_auc)

    plt.figure(1)
    plt.xlabel("FAP")
    plt.xscale("log")
    if logy == True:
        plt.yscale("log")
    plt.ylim(ylim, 1)
    plt.xlim(1e-5, 1)
    plt.ylabel("Efficiency")
    plt.legend()
    plt.grid()
    return fig, rocs


def get_fars(df, col, df_ref, col_ref, plot=False, logy=False):
    """
    Calculates the false positive probability(FPP) for the given pairs of 
    events and their ranking statistic by using the ranking statistics(or ROC)
    of the background injections as reference. This is done by interpolating
    the thresholds and FPP curve of the background injections at the values 
    of the ranking statistics of the pairs in hand.

    Parameters:
        df(pandas Dataframe): Dataframe containing the event pairs 
            ('img_0','img_1') and their ranking statistic, as columns, 
            for which the FPPs are to be calculated.
            
        col(str): name of Ranking statistic column for the pairs in hand.
        
        df_ref(pandas Dataframe): Dataframe containing the event 
            pairs('img_0','img_1'), labels('Lensing') and their ranking
            statistic as columns, for the background injections.
            
        col(str): name of Ranking statistic column for the background
            injections, which will be taken as reference.
            
        plot(bool):Return the Threshold v/s FPP plot, default: False.
        
        logy(float): Plot the yscale in log, default: False

    Returns:
        1-d numpy array of the false posiive probabilities for the pairs in hand.

    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        df_ref.Lensing.values, df_ref[col_ref]
    )
    fars = np.interp(
        df[col], np.flip(thresholds), np.flip(false_positive_rate)
    )
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    if plot == True:
        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, false_positive_rate, label=col_ref)
        plt.plot(df[col], fars, "x")
        for i in range(len(fars)):
            text = "pair %d : %E" % (i, fars[i])
            plt.text(df[col][i], fars[i], text)
            print(
                text
                + " id : "
                + str(df["img_0"][i])[:3]
                + "-"
                + str(df["img_1"][i])[:3]
            )
        plt.xlim(0, 1)
        plt.grid()
        plt.xlabel("Threshold")
        plt.ylabel("FAP")
        if logy == True:
            plt.yscale("log")
        plt.legend()
    return fars
