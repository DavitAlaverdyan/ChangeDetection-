import tensorflow.keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Layer, Input, Dropout, MaxPool2D, concatenate, BatchNormalization
import tensorflow as tf

class SiamUnetDiff(Layer):
    """SiamUnet_diff .

    This network is designed for Change Detection.

    Attributes:
        filters (int): Number of filters in the convolutional layers.
        rate (float): Dropout rate to be applied after each convolutional layer.
        pooling (bool): Whether to include max-pooling layers between stages.
    """

    def __init__(self, filters, rate, pooling=True):
        super(SiamUnetDiff, self).__init__()
        self.pooling = pooling
        self.filters = filters
        self.rate = rate
        self.pool = MaxPool2D(pool_size=(2, 2))

        # Stage 1
        self.c1 = Conv2D(self.filters, kernel_size=3, padding='same', activation='relu')
        self.c11 = Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.c111 = Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.drop1 = Dropout(self.rate)

        # Stage 2
        self.c2 = Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.c22 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.c222 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.drop2 = Dropout(self.rate)

        # Stage 3
        self.c3 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.c33 = Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.c333 = Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.c3333 = Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.drop3 = Dropout(self.rate)

        # Stage 4
        self.c4 = Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.c44 = Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.c444 = Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.c4444 = Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.drop4 = Dropout(self.rate)

        # Stage 5
        self.c5 = Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.c55 = Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.c555 = Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.drop5 = Dropout(self.rate)

        # Transpose convolutional layers for decoding
        self.ct1 = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')

        self.cv1 = Conv2DTranspose(256, kernel_size=3, padding='same', activation='relu')
        self.cv11 = Conv2DTranspose(256, kernel_size=3, padding='same', activation='relu')
        self.cv111 = Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu')
        self.dr1 = Dropout(self.rate)

        self.ct2 = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')

        self.cv2 = Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu')
        self.cv22 = Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu')
        self.cv222 = Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu')

        self.dr2 = Dropout(self.rate)

        self.ct3 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')

        self.cv3 = Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu')
        self.cv33 = Conv2DTranspose(32, kernel_size=3, padding='same', activation='relu')
        self.dr3 = Dropout(self.rate)

        self.ct4 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')

        self.cv4 = Conv2DTranspose(32, kernel_size=3, padding='same', activation='relu')
        self.cv44 = Conv2DTranspose(16, kernel_size=3, padding='same', activation='relu')
        self.dr4 = Dropout(self.rate)

        self.ct5 = Conv2DTranspose(16, kernel_size=3, strides=2, padding='same')

        self.cv5 = Conv2DTranspose(16, kernel_size=3, padding='same', activation='relu')
        self.cv55 = Conv2DTranspose(3, kernel_size=3, padding='same', activation='sigmoid')
        self.dr5 = Dropout(self.rate)

        self.pool = MaxPool2D(pool_size=(2, 2))

    def call(self, x1, x2):
        """Forward pass of the SiamUnetDiff model.

        Args:
            x1 (tf.Tensor): Input tensor for the first branch.
            x2 (tf.Tensor): Input tensor for the second branch.

        Returns:
            tf.Tensor: Output tensor representing the segmentation result.
        """
        # Stage 1
        x11 = self.bn1(self.drop1(self.c1(x1)))
        x11 = self.c11(x11)
        x11 = self.c111(x11)
        y11 = self.pool(x11)

        x21 = self.bn1(self.drop1(self.c1(x2)))
        x21 = self.c11(x21)
        x21 = self.c111(x21)
        y21 = self.pool(x21)

        # Stage 2
        x12 = self.bn2(self.drop2(self.c2(y11)))
        x12 = self.c22(x12)
        x12 = self.c222(x12)
        y12 = self.pool(x12
