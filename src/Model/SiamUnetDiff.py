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
        x11 = self.bn1(self.drop1(self.c1(x1)))
		x11 = self.c11(x11)
		x11 = self.c111(x11)
		y11 = self.pool(x11)

		x21 = self.bn1(self.drop1(self.c1(x2)))
		x21 = self.c11(x21)
		x21 = self.c111(x21)
		y21 = self.pool(x21)
		
		#stage 2
		x12 = self.bn2(self.drop2(self.c2(y11)))
		x12 = self.c22(x12)
		x12 = self.c222(x12)
		y12 = self.pool(x12) 
		
		x22 = self.bn2(self.drop2(self.c2(y21)))
		x22 = self.c22(x22)
		x22 = self.c222(x22)
		y22 = self.pool(x22)       
		
		#stage 3 
		x13 = self.bn3(self.drop3(self.c3(y12)))
		x13 = self.c33(x13)
		x13 = self.c333(x13)
		x13 = self.c3333(x13) 
		y13 = self.pool(x13)
		
		x23 = self.bn3(self.drop3(self.c3(y22)))
		x23 = self.c33(x23)
		x23 = self.c333(x23)
		x23 = self.c3333(x23) 
		y23 = self.pool(x23)
		
		#stage 4 
		x14 = self.bn4(self.c4(y13))
		x14 = self.c44(x14)
		x14 = self.c444(x14)
		x14 = self.c4444(x14)     
		y14 = self.pool(x14)
		
		x24 = self.bn4(self.c4(y23))
		x24 = self.c44(x24)
		x24 = self.c444(x24)
		x24 = self.c4444(x24) 
		y24 = self.pool(x24)
		
		#stage 5
		x15 = self.bn5(self.drop5(self.c5(y14)))
		x15 = self.c55(x15)
		x15 = self.c555(x15)
		y15 = self.pool(x15) 
		
		x25 = self.bn5(self.drop5(self.c5(y24)))
		x25 = self.c55(x25)
		x25 = self.c555(x25)
		y25 = self.pool(x25)   
		
		#stage 5d
		p1 = self.ct1(y15)
		p1 = self.cv1(concatenate([p1, tf.abs(x25 - x15)]))
		p1 = self.dr1(p1)
		p1 = self.cv11(p1)
		p1 = self.cv111(p1)
		
		#stage 4d
		p2 = self.ct2(p1)
		p2 = self.cv2(concatenate([p2, tf.abs(x24 - x14)]))
		p2 = self.dr2(p2)
		p2 = self.cv22(p2)
		p2 = self.cv222(p2)
		
		#stage 3d
		p3 = self.ct3(p2)
		p3 = self.cv3(concatenate([p3, tf.abs(x23 - x13)]))
		p3 = self.cv33(p3)
		
		#stage 2 d
		p4 = self.ct4(p3)
		p4 = self.cv4(concatenate([p4, tf.abs(x22 - x12)]))
		p4 = self.dr4(p4)
		p4 = self.cv44(p4)
		
		#stage 1 d
		p5 = self.ct5(p4)
		p5 = self.cv5(concatenate([p5, tf.abs(x21 - x11)]))
		p5 = self.dr5(p5)
		p5 = self.cv55(p5)
		
		
		return p5