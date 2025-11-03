import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers ,losses
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.ops.gen_batch_ops import batch

url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' #下载地址
dataset = tf.keras.utils.get_file('aclImdb_v1',url,
                                  untar=True , cache_dir='.',
                                  cache_subdir='')  #存储
dataset_dir = os.path.join(os.path.dirname(dataset),'aclImdb')
#浏览数据结构
# print(os.listdir(dataset_dir))
train_dir = os.path.join(dataset_dir,'train')
# print(os.listdir(train_dir))
# #查看其中一条评论
# sample_file = os.path.join(train_dir,'pos/1181_9.txt')
# with open(sample_file) as f:
#     print(f.read())
#移除imdb数据集中的unsup
remove_dir = os.path.join(train_dir,'unsup')
shutil.rmtree(remove_dir)
#拆分数据集
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,#按80:20拆分训练集数据来创建测试集
    subset='training',
    seed=seed
)
# print(raw_train_ds)
# #查看标签为0或1与正负电影评论的对应关系
# print('Label 0 corresponds to' , raw_train_ds.class_names[0])
# print('Label 1 corresponds to' , raw_train_ds.class_names[1])
#创建验证数据集和测试数据集，这里使用指定随机种子，如果不指定随机种子就传递shuffle=False
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)
# print(raw_val_ds)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size,
)
# print(raw_test_ds)
#准备用于训练的数据集，使用tf.keras.layers.TextVectorization 层对数据进行标准化、词例化和向量化。
#TextVectorization 层（默认情况下会将文本转换为小写并去除标点符号，但不会去除 HTML）
#所以这里自定义标准化函数来移除HTML
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
max_features = 10000
sequence_length = 250
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)
#调用 adapt 以使预处理层的状态适合数据集
train_text = raw_train_ds.map(lambda x , y : x)
vectorize_layer.adapt(train_text)
#创建函数查看使用该层预处理一些数据的结果
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
text_batch, label_batch = next(iter(raw_train_ds))
frist_review, frist_label = text_batch[0], label_batch[0]
# print('Review', frist_review)
# print('Label', raw_train_ds.class_names[frist_label])
# print('Vectorized review', vectorize_text(frist_review,frist_label))
# print('1287--->', vectorize_layer.get_vocabulary()[1287])
# print('38--->', vectorize_layer.get_vocabulary()[38])
# print('Vocabulary size:{}'.format(len(vectorize_layer.get_vocabulary())))

#将之前创建的 TextVectorization 层应用于训练数据集、验证数据集和测试数据集。
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

#配置数据集以提高性能
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(20000).cache().prefetch(buffer_size=AUTOTUNE) #.shuffle(20000)随机打乱20000数据，.cache用来将数据存到内存中，prefetch启用并行处理
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#创建模型
embedding_dim = 16

# model = tf.keras.Sequential([
#     layers.Embedding(max_features + 1, embedding_dim,
#                      input_length = sequence_length,
#                      trainable = True,
#                      mask_zero = True,),
#
#     layers.LSTM(128,return_sequences=False),#LSTM层：提取序列特征
#     layers.Dropout(0.2),
#     layers.Reshape((-1,128)),#由于GlobalAveragePooling1D()需要数据是3维的，这里增加一下维度，不过其实也可以使用其他池化策略，如Flatten（）
#     layers.GlobalAveragePooling1D(),#GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入
#     layers.Dropout(0.2),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(1, activation='sigmoid')
# ])

model = tf.keras.Sequential([

    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

model.summary()

#编译模型
model.compile(optimizer='adam',
              loss = losses.BinaryCrossentropy(from_logits=True),#binary_crossentropy 更适合处理概率——它能够度量概率分布之间的“距离”
              metrics= tf.metrics.BinaryAccuracy(threshold=0.0))

#使用早停机制
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=5,
                                              restore_best_weights=True)

#训练模型
history = model.fit(train_ds,
                    epochs=60,
                    validation_data=val_ds,
                    callbacks=[early_stop],
                    verbose=1)

#评估模型
loss, accuracy = model.evaluate(test_ds)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#创建准确率和损失随时间变化的图表

history_dict = history.history
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

model.save('imdb2.h5')