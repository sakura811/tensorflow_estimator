# -*- coding: utf-8 -*-
'''
can run
'''

import os
import pandas as pd
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 自定义模型函数
def my_model_fn(features, labels, mode, params):
    # 输入层,feature_columns对应Classifier(feature_columns=...)
    # net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'])

    # 隐藏层,hidden_units对应Classifier(unit=[10,10])，2个各含10节点的隐藏层
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # 输出层，n_classes对应3种鸢尾花
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # 预测
    predicted_classes = tf.argmax(logits, 1)  # 预测的结果中最大值即种类
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # 拼成列表[[3],[2]]格式
            'probabilities': tf.nn.softmax(logits),  # 把[-1.3,2.6,-0.9]规则化到0~1范围,表示可能性
            'logits': logits,  # [-1.3,2.6,-0.9]
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 损失函数
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # 训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)  # 用它优化损失函数，达到损失最少精度最高
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)
        # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  # 执行优化！
        train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # 评价
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')  # 计算精度
    metrics = {'accuracy': accuracy}  # 返回格式
    tf.summary.scalar('accuracy', accuracy[1])  # 仅为了后面图表统计使用
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


FUTURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


# 格式化数据文件的目录地址
dir_path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(dir_path, 'iris_training.csv')
test_path = os.path.join(dir_path, 'iris_test.csv')

# 载入训练数据
train = pd.read_csv(train_path, names=FUTURES, header=0)
train_x, train_y = train, train.pop('Species')
print('train_x: \n', train_x)
print('train_y: \n', train_y)

# 载入测试数据
test = pd.read_csv(test_path, names=FUTURES, header=0)
test_x, test_y = test, test.pop('Species')
print('test_x: \n', test_x)
print('test_y: \n', test_y)

# 设定特征值的名称
feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# tf.logging.set_verbosity(tf.logging.INFO)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

models_path=os.path.join(dir_path,'mymodels/')

#创建自定义分类器
classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir=models_path,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        })


# 针对训练的喂食函数
def train_input_fn(features, labels, batch_size):  # model_fn
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)  # 每次随机调整数据顺序
    # dataset = dataset.make_one_shot_iterator().get_next()  # dataset.make_one_shot_iterator()在分发策略/高级库之外很有用，例如，如果您正在使用低级库或调试/测试数据集。例如，您可以像这样迭代所有数据集的元素：
    print('train_input_fn dataset: ', dataset)
    return dataset


# 设定仅输出警告提示，可改为INFO
# tf.logging.set_verbosity(tf.logging.WARN)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

# 开始训练模型！
batch_size = 100
classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=1000)


# 针对测试的喂食函数
def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.make_one_shot_iterator().get_next()
    return dataset

# 评估我们训练出来的模型质量
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, batch_size))
print(eval_result)


# 支持100次循环对新数据 进行分类预测
for i in range(0, 2):
    print('i = ', i)
    print('\nPlease enter features: SepalLength,SepalWidth,PetalLength,PetalWidth')
    # a, b, c, d = map(float, input().split(','))  # 捕获用户输入的数字
    a = 1
    b = 2
    c = 3
    d = 4
    predict_x = {
        'SepalLength': [a],
        'SepalWidth': [b],
        'PetalLength': [c],
        'PetalWidth': [d],
    }

    # 进行预测
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x,
                                       labels=[0],
                                       batch_size=batch_size))

    # 预测结果是数组，尽管实际我们只有一个
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(SPECIES[class_id], 100 * probability)
