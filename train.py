# -*- coding: utf-8 -*-

import os
import numpy
from models import *
import random
import sys
import time
import datetime
from tqdm import tqdm
import tensorflow as tf
import glob
import json
from functools import wraps
from tensorflow import contrib
from sklearn.metrics import roc_auc_score
from sklearn import metrics

autograph = contrib.autograph
rootPath = os.getcwd()
sys.path.append(rootPath)
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#----------------------------------params----------------------------------------------
tf.app.flags.DEFINE_string('f','' ,'kernel')
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir", './result_data/', "data dir")
tf.app.flags.DEFINE_string("hdfs_dir", 'hdfs://amazon', "hdfs dir")

tf.app.flags.DEFINE_integer("maxlen", 50, "maxlen")
tf.app.flags.DEFINE_integer("uselen", 50, "uselen")
tf.app.flags.DEFINE_integer("shuffle_buffer_size", 50000, "shuffle_buffer_size")
tf.app.flags.DEFINE_integer("user_size",15362620, "user_size")#user
tf.app.flags.DEFINE_integer("item_size",2930452, "item_size")#item
tf.app.flags.DEFINE_integer("cate_size",570000, "cate_size")#item

tf.app.flags.DEFINE_integer("embedding_dim", 16, "embedding_dim")
tf.app.flags.DEFINE_integer("use_aux_loss", 0, "use_aux_loss")
tf.app.flags.DEFINE_integer("model_dim", 64, "model_dim")
tf.app.flags.DEFINE_float("lr", 0.0001, "dnn learning rate")
tf.app.flags.DEFINE_float("dropout_rate", 0.0, "dropout_rate")#droup out
tf.app.flags.DEFINE_float("l2", 0.003, "l2")
tf.app.flags.DEFINE_integer("num_heads", 8, "num_heads")
tf.app.flags.DEFINE_string("deep_layers", '128,64', "deep_layers")
#
tf.app.flags.DEFINE_integer("train_epochs",1, "train epochs")#
tf.app.flags.DEFINE_integer("log_steps", 5, "log_steps")
tf.app.flags.DEFINE_integer("cpu_num", 9, "cpu_num")
tf.app.flags.DEFINE_integer("batch_size", 1024, "batch_size")
tf.app.flags.DEFINE_integer("batch_size_test", 2048, "batch_size_test")
tf.app.flags.DEFINE_integer("eval_per_num", 10, "eval_per_num")#
tf.app.flags.DEFINE_integer("is_part_test",1, "is_part_test")#
tf.app.flags.DEFINE_integer("batch_size_total_test", 10, "batch_size_total_test")#
tf.app.flags.DEFINE_integer("local_train",1, "local_train")#
#
tf.app.flags.DEFINE_string("data_type", 'elec', "data_type")#book,cloth,elec
tf.app.flags.DEFINE_string("test_data", '[11,12]', "test_data")
#
tf.app.flags.DEFINE_string("model_type", 'lr', "model_type")#cmin,din,dien,lr,mlp

def input_fn(filenames, is_train = True, batch_size = FLAGS.batch_size):
    print('#' * 100)
    print('loading data: ', filenames[0])
    def decode_fn(line):
        items = tf.string_split([line], '\t')
        
        label = tf.string_to_number(items.values[0], out_type=tf.float32)
        user_id = tf.string_to_number(items.values[1], out_type=tf.int32)
        item_id = tf.string_to_number(items.values[2], out_type=tf.int32)
        cate_id = tf.string_to_number(items.values[3], out_type=tf.int32)
        
        week = tf.string_to_number(items.values[4], out_type=tf.int32)
        month = tf.string_to_number(items.values[5], out_type=tf.int32)
        
        click_seq = tf.string_split([items.values[6]], ',')
        click_seq = tf.string_to_number(click_seq.values, out_type=tf.int32)
        cateid_seq = tf.string_split([items.values[7]], ',')
        cateid_seq = tf.string_to_number(cateid_seq.values, out_type=tf.int32)
        convert_mask = tf.string_split([items.values[8]], ',')
        convert_mask = tf.string_to_number(convert_mask.values, out_type=tf.float32)
        weeks = tf.string_split([items.values[9]], ',')
        weeks = tf.string_to_number(weeks.values, out_type=tf.int32)
        months = tf.string_split([items.values[10]], ',')
        months = tf.string_to_number(months.values, out_type=tf.int32)
        diff_months = tf.string_split([items.values[11]], ',')
        diff_months = tf.string_to_number(diff_months.values, out_type=tf.int32)
        diff_days = tf.string_split([items.values[12]], ',')
        diff_days = tf.string_to_number(diff_days.values, out_type=tf.int32)
        
        click_neg_seq = tf.string_split([items.values[14]], ',')
        click_neg_seq = tf.string_to_number(click_neg_seq.values, out_type=tf.int32)
        cateid_neg_seq = tf.string_split([items.values[15]], ',')
        cateid_neg_seq = tf.string_to_number(cateid_neg_seq.values, out_type=tf.int32)
        
        return {"label": label, "user_id": user_id, "item_id": item_id, "cate_id": cate_id, "week": week, 
                "month":month,"click_seq":click_seq,"cateid_seq":cateid_seq, "convert_mask":convert_mask,
               "weeks":weeks,"months":months,"diff_months":diff_months,"diff_days":diff_days,
               "click_neg_seq":click_neg_seq, "cateid_neg_seq":cateid_neg_seq}

    dataset = tf.data.TextLineDataset(filenames)#
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)#
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=decode_fn, batch_size=batch_size,
                                                          drop_remainder=True,
                                                          num_parallel_batches=FLAGS.cpu_num))
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset
#-----------------------------
def test(sess, test_file, model, n, step, is_part_test = FLAGS.is_part_test):
    def c_round(x, s = 0.5):
        xx = x // 1
        if x - xx < s:
            return int(xx)
        else:
            return int(xx + 1)
    batch_size = FLAGS.batch_size_test
    dataset_test = input_fn(test_file, is_train = False, batch_size = batch_size)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()
    sess.run(iterator_test.initializer)
    test_step = 0
    test_loss_sum = 0
    y_true = []#
    y_scores = []#
    y_loss = 0#loss
    try:#---------------
        while True:
            test_step += 1
            test_input = sess.run(next_element_test)
            prob, loss = model.evaluate(sess, test_input)
            y_scores += prob[:, 0].tolist()
            y_true += test_input["label"].tolist()
            y_loss += loss
            if test_step % FLAGS.log_steps == 0:
                #print("sample: prob=" + str(prob[:10, 0].tolist()) + ", true=" + str(test_input["label"].tolist()[:10]))
                print("【test】:mean_loss = %.6f\tbatch = %d\tstep = %d\tepochs = %d" \
                              % (y_loss / test_step, batch_size, test_step, n+1))
                sys.stdout.flush()
            if test_step >= FLAGS.batch_size_total_test and is_part_test == 1:#
                auc = roc_auc_score(y_true, y_scores)
                print('#' * 100)
                for s in [0.5]:
                    acc = metrics.accuracy_score([c_round(x,s) for x in y_true], [c_round(x,s) for x in y_scores])
                    f1_score = metrics.f1_score([c_round(x,s) for x in y_true], [c_round(x,s) for x in y_scores])
                    print('【END】：auc =  %.6f \t acc =  %.6f \t f1 =  %.6f \t s=%.2f \tloss = %.6f' % (auc,acc,f1_score,s, y_loss/test_step))
                print('#' * 110)
                sys.stdout.flush()
                break
    except tf.errors.OutOfRangeError:#-----------------  
        auc = roc_auc_score(y_true, y_scores)
        print('#' * 100)
        for s in [0.5]:
            acc = metrics.accuracy_score([c_round(x,s) for x in y_true], [c_round(x,s) for x in y_scores])
            f1_score = metrics.f1_score([c_round(x,s) for x in y_true], [c_round(x,s) for x in y_scores])
            print('【END】：auc =  %.6f \t acc =  %.6f \t f1 =  %.6f \t s=%.2f \tloss = %.6f' % (auc,acc,f1_score,s, y_loss/test_step))
        print('#' * 110)
        sys.stdout.flush()
    return acc, auc, f1_score

def main():
    #------------------------------------------------
    if FLAGS.local_train:
        train_path = glob.glob("%s/small_subset_book/train*" % FLAGS.data_dir)
        test_path = glob.glob("%s/small_subset_book/test*" % FLAGS.data_dir)
    else:
        train_path = []
        test_path = []
        for i in range(12):
            if i+1 not in eval(FLAGS.test_data):
                train_path += ["%s/aux_%s/part-2017%02d" % (FLAGS.hdfs_dir,FLAGS.data_type,i+1)]
            else:
                test_path += ["%s/aux_%s/part-2017%02d" % (FLAGS.hdfs_dir,FLAGS.data_type,i+1)]
        random.shuffle(train_path)
        random.shuffle(test_path)   
            
    #---------------------model params---------------
    model_params = {
        "maxlen": FLAGS.maxlen,
        "uselen": FLAGS.uselen,
        "embedding_dim": FLAGS.embedding_dim,
        "model_dim": FLAGS.model_dim,
        "lr": FLAGS.lr,
        "dropout_rate":FLAGS.dropout_rate,
        "user_size":FLAGS.user_size,
        "item_size":FLAGS.item_size, 
        "cate_size":FLAGS.cate_size, 
        "model_type":FLAGS.model_type,
        "l2":FLAGS.l2,
        "use_aux_loss":FLAGS.use_aux_loss,
        "num_heads":FLAGS.num_heads,
        "dropout_rate":FLAGS.dropout_rate,
        "deep_layers":FLAGS.deep_layers
    }
    #初始化sess的config
    config = tf.ConfigProto(device_count={"CPU": FLAGS.cpu_num}, inter_op_parallelism_threads=FLAGS.cpu_num,
                            intra_op_parallelism_threads=FLAGS.cpu_num, log_device_placement=False, allow_soft_placement=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    #--------------------training-------------------
    with tf.Session(config=config) as sess:
        model = Model(model_params)
        dataset = input_fn(train_path, is_train = True)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        sys.stdout.flush()
        writer = tf.summary.FileWriter("tensorboard", sess.graph)
        auc = 0
        step = 0
        for n in range(FLAGS.train_epochs):
            sess.run(iterator.initializer)#
            #train
            loss_sum = 0
            input_time = 0
            model_time = 0
            try:
                 while True:
                    st = time.time()
                    model_input = sess.run(next_element)#
                    input_t = time.time()-st#
                    if input_time == 0:
                        input_time = input_t
                    else:
                        input_time = (input_time + input_t) / 2
                    loss,t = model.train(sess, model_input)#
                    step += 1#step
                    model_t = time.time() - st - input_t
                    if model_time == 0:#
                        model_time = model_t
                    else:
                        model_time = (model_time + model_t) / 2
                    loss_sum += loss
                    if step % FLAGS.log_steps == 0:
                        writer.add_summary(summary=t, global_step=step)
                        mean_loss = loss_sum / FLAGS.log_steps
                        print("【train】:mean_loss = %.6f\tbatch = %d\tinput = %.3f sec\ttrain = %.3f sec\tstep = %d\tepochs = %d" \
                              % (mean_loss, FLAGS.batch_size, input_time, model_time, step, n+1))
                        sys.stdout.flush()
                        input_time = 0
                        model_time = 0
                        loss_sum = 0
                    if step % FLAGS.eval_per_num == 0 and FLAGS.eval_per_num != -1:
                        acc, auc, f1_score = test(sess, test_path, model, n, step) 
                        if FLAGS.local_train:
                            return
            except tf.errors.OutOfRangeError:
                if n ==FLAGS.train_epochs -1:#
                    acc, auc, f1_score = test(sess, test_path, model, n, step)    
        sys.stdout.flush()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main()