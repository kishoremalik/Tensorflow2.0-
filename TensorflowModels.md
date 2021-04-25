Models Type
1) Sequential
2) Functional
3) SubClass 


1)Sequential



Functional  Model for CNN 
----------------------------
<pre>
batch_size=64
WEIGHT_DECAY=0.001
Learning_rate=0.001


inputs=Input(shape=(32,32,3))
x=layers.Conv2D(32,3,padding='same',activation='relu',kernel_regularizer=regularizers.L2(WEIGHT_DECAY))(inputs)
#x=layers.Dropout(0.5)(x)
x=layers.BatchNormalization()(x)
x=layers.MaxPooling2D()(x)
x=layers.Conv2D(64,3,activation='relu',kernel_regularizer=regularizers.L2(WEIGHT_DECAY))(x)
#x=layers.Dropout(0.5)(x)
x=layers.BatchNormalization()(x)
x=layers.MaxPooling2D()(x)
x=layers.Flatten()(x)
x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.L2(WEIGHT_DECAY))(x)
x=layers.Dropout(0.5)(x)
x=layers.Dense(10)(x)

model=Model(inputs=inputs,outputs=x)
model.compile(optimizer=keras.optimizers.Adam(lr=Learning_rate),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
</pre>

Sub Class Model 
--------------------------------
<pre>

class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same",kernel_regularizer=regularizers.L2(0.001))
        self.bn = layers.BatchNormalization()
        self.drop=layers.Dropout(0.4)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


model = keras.Sequential(
    [CNNBlock(32), CNNBlock(64), CNNBlock(128), layers.Flatten(), layers.Dense(10),]
)


class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.cnn1 = CNNBlock(channels[0], 3)
        self.cnn2 = CNNBlock(channels[1], 3)
        self.cnn3 = CNNBlock(channels[2], 3)
        self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], 3, padding="same")

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x + self.identity_mapping(input_tensor), training=training,)
        x = self.pooling(x)
        return x


class ResNet_Like(keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x, training=training)
        x = self.classifier(x)
        return x

    def model(self):
        x = keras.Input(shape=(32, 32, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))


batch_size=64
WEIGHT_DECAY=0.001
Learning_rate=0.001

model = ResNet_Like().model()
base_input = model.layers[0].input
base_output = model.layers[2].output
output = layers.Dense(10)(layers.Flatten()(base_output))
model = keras.Model(base_input, output)
model.compile(optimizer=keras.optimizers.Adam(lr=Learning_rate),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
</pre>
