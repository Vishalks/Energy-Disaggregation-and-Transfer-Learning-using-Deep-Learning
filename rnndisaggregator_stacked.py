from __future__ import print_function, division
import random
import sys

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout, GRU
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.models import Model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class RNNDisaggregator(Disaggregator):
    '''Attempt to create a RNN Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "LSTM"
        self.mmax = None
        self.MIN_CHUNK_LENGTH = 100
        self.model = self._create_model()
        self.stacked_model = None

    def train(self, mains, meter, epochs=1, batch_size=128, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        batch_size : size of batch used for training
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = mainchunk.max()

        while(run):
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)

            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False

    # fit a stacked model
    def fit_stacked_model(self, inputX, inputy, epochs):
        # prepare input data
        X = [inputX for _ in range(len(self.stacked_model.input))]
        # encode output data
        #inputy_enc = to_categorical(inputy)
        # fit model
        self.stacked_model.fit(X, inputy, epochs=epochs, verbose=0)

    # define stacked model from multiple member input models
    def define_stacked_model(self, members):
        # update all layers in all models to not be trainable
        for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unique layer name' issue
                layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
        # define multi-headed input
        ensemble_visible = [model.input for model in members]
        # concatenate merge output from each model
        ensemble_outputs = [model.output for model in members]
        merge = concatenate(ensemble_outputs)
        hidden = Dense(64, activation='relu')(merge)
        output = Dense(1, activation='relu')(hidden)
        model = Model(inputs=ensemble_visible, outputs=output)
        model.compile(loss='mse', optimizer='adam')
        return model

    # make a prediction with a stacked model
    def predict_stacked_model(self, inputX, batch_size):
        # prepare input data
        X = [inputX for _ in range(len(self.stacked_model.input))]
        # make prediction
        return self.stacked_model.predict(X, verbose=0, batch_size=batch_size)

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        batch_size : size of batch used for training
        '''

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = np.array(mainchunk[ix])
        meterchunk = np.array(meterchunk[ix])

        mainchunk = np.reshape(mainchunk, (mainchunk.shape[0],1,1))

        #self.model.fit(mainchunk, meterchunk, epochs=epochs, batch_size=batch_size, shuffle=True)
        for i in range(len(self.model)):
            self.model[i].fit(mainchunk, meterchunk, batch_size=batch_size, epochs=epochs, shuffle=True)
            #filename = 'models/model_' + str(i + 1) + '.h5'
            #model.save(filename)
            #print('>Saved %s' % filename)

        self.stacked_model = self.define_stacked_model(self.model)
        self.fit_stacked_model(mainchunk, meterchunk, epochs)


    def train_across_buildings(self, mainlist, meterlist, epochs=1, batch_size=128, **load_kwargs):
        '''Train using data from multiple buildings

        Parameters
        ----------
        mainlist : a list of nilmtk.ElecMeter objects for the aggregate data of each building
        meterlist : a list of nilmtk.ElecMeter objects for the meter data of each building
        batch_size : size of batch used for training
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        assert len(mainlist) == len(meterlist), "Number of main and meter channels should be equal"
        num_meters = len(mainlist)

        mainps = [None] * num_meters
        meterps = [None] * num_meters
        mainchunks = [None] * num_meters
        meterchunks = [None] * num_meters

        # Get generators of timeseries
        for i,m in enumerate(mainlist):
            mainps[i] = m.power_series(**load_kwargs)

        for i,m in enumerate(meterlist):
            meterps[i] = m.power_series(**load_kwargs)

        # Get a chunk of data
        for i in range(num_meters):
            mainchunks[i] = next(mainps[i])
            meterchunks[i] = next(meterps[i])
        if self.mmax == None:
            self.mmax = max([m.max() for m in mainchunks])


        run = True
        while(run):
            # Normalize and train
            mainchunks = [self._normalize(m, self.mmax) for m in mainchunks]
            meterchunks = [self._normalize(m, self.mmax) for m in meterchunks]

            self.train_across_buildings_chunk(mainchunks, meterchunks, epochs, batch_size)

            # If more chunks, repeat
            try:
                for i in range(num_meters):
                    mainchunks[i] = next(mainps[i])
                    meterchunks[i] = next(meterps[i])
            except:
                run = False

    def train_across_buildings_chunk(self, mainchunks, meterchunks, epochs, batch_size):
        '''Train using only one chunk of data. This chunk consists of data from
        all buildings.

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        batch_size : size of batch used for training
        '''
        num_meters = len(mainchunks)
        batch_size = int(batch_size/num_meters)
        num_of_batches = [None] * num_meters

        # Find common parts of timeseries
        for i in range(num_meters):
            mainchunks[i].fillna(0, inplace=True)
            meterchunks[i].fillna(0, inplace=True)
            ix = mainchunks[i].index.intersection(meterchunks[i].index)
            m1 = mainchunks[i]
            m2 = meterchunks[i]
            mainchunks[i] = m1[ix]
            meterchunks[i] = m2[ix]

            num_of_batches[i] = int(len(ix)/batch_size) - 1

        for e in range(epochs): # Iterate for every epoch
            print(e)
            batch_indexes = list(range(min(num_of_batches)))
            random.shuffle(batch_indexes)

            for bi, b in enumerate(batch_indexes): # Iterate for every batch
                print("Batch {} of {}".format(bi,num_of_batches), end="\r")
                sys.stdout.flush()
                X_batch = np.empty((batch_size*num_meters, 1, 1))
                Y_batch = np.empty((batch_size*num_meters, 1))

                # Create a batch out of data from all buildings
                for i in range(num_meters):
                    mainpart = mainchunks[i]
                    meterpart = meterchunks[i]
                    mainpart = mainpart[b*batch_size:(b+1)*batch_size]
                    meterpart = meterpart[b*batch_size:(b+1)*batch_size]
                    X = np.reshape(mainpart, (batch_size, 1, 1))
                    Y = np.reshape(meterpart, (batch_size, 1))

                    X_batch[i*batch_size:(i+1)*batch_size] = np.array(X)
                    Y_batch[i*batch_size:(i+1)*batch_size] = np.array(Y)

                # Shuffle data
                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]

                # Train model
                self.model.train_on_batch(X_batch, Y_batch)
            print("\n")

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : a nilmtk.ElecMeter of aggregate data
        meter_metadata: a nilmtk.ElecMeter of the observed meter used for storing the metadata
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series of aggregate data
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array(mains)
        X_batch = np.reshape(X_batch, (X_batch.shape[0],1,1))

        #pred = self.model.predict(X_batch, batch_size=128)
        pred = self.predict_stacked_model(X_batch, 128)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[:len(X_batch)], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    def _create_model(self):
        '''Creates the RNN module described in the paper
        '''
        model = []

        model1 = Sequential()

        # 1D Conv
        model1.add(Conv1D(16, 4, activation="linear", input_shape=(1,1), padding="same", strides=1))

        #Bi-directional LSTMs
        model1.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
        model1.add(Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))

        # Fully Connected Layers
        model1.add(Dense(128, activation='tanh'))
        model1.add(Dense(1, activation='linear'))

        model1.compile(loss='mse', optimizer='adam')
        model.append(model1)


        model2 = Sequential()

        # 1D Conv
        model2.add(Conv1D(16, 4, activation="relu", padding="same", strides=1, input_shape=(1,1)))
        model2.add(Conv1D(8, 4, activation="relu", padding="same", strides=1))

        # Bi-directional LSTMs
        model2.add(Bidirectional(GRU(64, return_sequences=True, stateful=False), merge_mode='concat'))
        model2.add(Bidirectional(GRU(128, return_sequences=False, stateful=False), merge_mode='concat'))

        # Fully Connected Layers
        model2.add(Dense(64, activation='relu'))
        model2.add(Dense(1, activation='linear'))

        model2.compile(loss='mse', optimizer='adam')
        model.append(model2)

        return model
