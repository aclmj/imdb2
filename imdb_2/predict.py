from tensorflow.python.keras.saving.save import load_model

model = load_model('imdb2')
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]
model.predict(examples)

# import tensorflow as tf
# from tensorflow.keras.models import load_model
#
# # 加载模型
# model = load_model('imdb2.h5')
#
# # 打印模型结构
# print("模型结构:")
# model.summary()
#
# # 检查输入层的信息
# print("\n输入层信息:")
# print(f"输入形状: {model.input_shape}")
# print(f"输入数据类型: {model.input_dtype}")