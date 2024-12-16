import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class TrussNNet():
    def __init__(self, game, args):
        # game params
        self.board_y, self.board_x = game.getBoardSize() #daq -> change: x,y -> y,x
        self.action_size = game.getActionSize() 
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        #h_conv1 = Activation('relu')(Conv2D(args.num_channels, 8, padding='same', use_bias=False)(x_image))       # batch_size  x board_x x board_y x num_channels
        #h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 8, padding='same', use_bias=False)(x_image)))   
        h_conv2 = Activation('relu')(Conv2D(args.num_channels, 5, padding='same', use_bias=False)(x_image))        # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv2))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        #h_conv4 = Activation('relu')(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv3)       
        #s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        #s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        s_fc1 = (Activation('relu')(Dense(1024)(h_conv4_flat)))  # batch_size x 1024
        s_fc2 = (Activation('relu')(Dense(512)(s_fc1)))          # batch_size x 1024
        #self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        #self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1
        self.v = Dense(1, name='v')(s_fc2)  

        #self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model = Model(inputs=self.input_boards, outputs=[self.v])
        #self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        self.model.compile(loss=['mean_squared_error'], optimizer=Adam(args.lr))
