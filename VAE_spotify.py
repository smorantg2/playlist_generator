from keras.models import Model
from keras.layers import Input, Lambda, LeakyReLU, Dropout
from keras.layers import Dense, BatchNormalization
from keras import backend as K

def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape = K.shape(mu), mean = 0, stddev = 1.)
    return mu + K.exp(log_var / 2) * epsilon

def vae_r_loss(y_true, y_pred):
    r_loss = K.mean(K.square(y_true - y_pred), axis = 1)
    return r_loss_factor * r_loss

def vae_kl_loss(y_true, y_pred):
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis = 1)
    return kl_loss

def vae_loss(y_true, y_pred):
    r_loss = vae_r_loss(y_true, y_pred)
    kl_loss = vae_kl_loss(y_true, y_pred)
    return r_loss + kl_loss


# ENCODER
size = 128
activation = "relu"
INPUT_DIM = 9

encoder_input = Input(shape=(INPUT_DIM,))
x = encoder_input
x = BatchNormalization()(x)
x = Dense(size)(x)
x = LeakyReLU()(x)
x = Dense(size)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(size)(x)
x = LeakyReLU()(x)
x = Dense(size)(x)
x = LeakyReLU()(x)
x = Dense(size)(x)
x = Dropout(0.2)(x)
mu = Dense(2)(x)
log_var = Dense(2)(x)

encoder_mu_log_var = Model(inputs=encoder_input, outputs=(mu,log_var))

encoder_output = Lambda(sampling, name = "encoder_output")([mu, log_var])

encoder = Model(inputs = encoder_input, outputs = encoder_output)

# encoder.summary()

# DECODER

input_decoder = Input(shape=(2,))
x = Dense(size)(input_decoder)
x = LeakyReLU()(x)
x = Dense(size)(x)
x = LeakyReLU()(x)
x = Dense(size)(x)
x = LeakyReLU()(x)
x = Dense(size)(x)
x = LeakyReLU()(x)
x = Dense(size)(x)
decoder_output = Dense(INPUT_DIM, activation = "sigmoid")(x)


decoder = Model(inputs=input_decoder, outputs=decoder_output)

# decoder.summary()

# MODEL

r_loss_factor = 1000

model_input = encoder_input
model_output = decoder(encoder_output)

model = Model(model_input, model_output)

model.compile(optimizer="rmsprop", loss=[vae_loss], metrics = [vae_r_loss, vae_kl_loss])

# from keras.models import load_weights

model.load_weights('VAE_spoty.h5')
