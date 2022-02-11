# -*- coding: utf-8 -*-

import tensorflow as tf
from model_parts import *
from rnn import dynamic_rnn

class Model(object):
    def __init__(self, params):
        #参数列表
        self.lr = params['lr']
        self.dropout_rate = params['dropout_rate']
        self.embedding_dim = params['embedding_dim']
        self.max_len = params['maxlen']
        self.user_size = params['user_size']
        self.item_size = params['item_size']
        self.cate_size = params['cate_size']
        self.model_type = params['model_type']
        self.l2 = params['l2']
        self.num_heads = params["num_heads"]
        self.deep_layers = params["deep_layers"]
        self.dnn_num1 = int(self.deep_layers.split(",")[0])
        self.dnn_num2 = int(self.deep_layers.split(",")[1])
        self.seq_len = params['uselen']
        self.model_dim = params['model_dim']
        self.use_aux_loss = params['use_aux_loss']
        
        #数据placeholder
        self.is_train = tf.placeholder_with_default(False, (), 'is_train')
        self.label = tf.placeholder(tf.float32, [None, ], name='label')#batch_size,
        
        self.user_id = tf.placeholder(tf.int32, [None, ], name='user_id')#batch_size,
        self.item_id = tf.placeholder(tf.int32, [None, ], name='item_id')#batch_size ,
        self.cate_id = tf.placeholder(tf.int32, [None, ], name='cate_id')#batch_size ,
        self.week = tf.placeholder(tf.int32, [None, ], name='week')#batch_size,
        self.month = tf.placeholder(tf.int32, [None, ], name='month')#batch_size,
        
        self.click_seq = tf.placeholder(tf.int32, [None, self.max_len], name='click_seq')#batch_size *  50
        self.cateid_seq = tf.placeholder(tf.int32, [None, self.max_len], name='cateid_seq')#batch_size *  50
        self.convert_mask = tf.placeholder(tf.float32, [None, self.max_len], name='convert_mask')#batch_size *  50
        self.weeks = tf.placeholder(tf.int32, [None, self.max_len], name='weeks')#batch_size *  50
        self.months = tf.placeholder(tf.int32, [None, self.max_len], name='months')#batch_size *  50
        self.diff_months = tf.placeholder(tf.int32, [None, self.max_len], name='diff_months')#batch_size *  50
        self.diff_days = tf.placeholder(tf.int32, [None, self.max_len], name='diff_days')#batch_size *  50
        self.click_neg_seq = tf.placeholder(tf.int32, [None, self.max_len], name='click_neg_seq')#batch_size *  50
        self.cateid_neg_seq = tf.placeholder(tf.int32, [None, self.max_len], name='cateid_neg_seq')#batch_size *  50
        #构建模型
        self.forward()
        
    def forward(self):
        '''前馈网络
        '''
        
        with tf.name_scope('Embedding_layer'):
            #length
            _, click_seq = tf.split(self.click_seq, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            _, cateid_seq = tf.split(self.cateid_seq, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            _, convert_mask = tf.split(self.convert_mask, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            _, weeks = tf.split(self.weeks, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            _, months = tf.split(self.months, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            _, diff_months = tf.split(self.diff_months, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            _, diff_days = tf.split(self.diff_days, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            if self.use_aux_loss:
                _, click_neg_seq = tf.split(self.click_neg_seq, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
                _, cateid_neg_seq = tf.split(self.cateid_neg_seq, [self.max_len-self.seq_len, self.seq_len], 1)#batch_size *  seq_len
            
            # user id embedding
            user_id_var = tf.get_variable("user_id_var", [self.user_size, self.embedding_dim])
            user_id_embedding = tf.nn.embedding_lookup(user_id_var, self.user_id)#batch_size  * embedding_dim
            # item id embedding
            item_id_var = tf.get_variable("item_id_var", [self.item_size, self.embedding_dim])
            item_id_embedding = tf.nn.embedding_lookup(item_id_var, self.item_id)#batch_size  * embedding_dim
            click_seq_embedding = tf.nn.embedding_lookup(item_id_var, click_seq)#batch_size * seq_len * embedding_dim
            # cate id embedding
            cate_id_var = tf.get_variable("cate_id_var", [self.cate_size, self.embedding_dim])
            cate_id_embedding = tf.nn.embedding_lookup(cate_id_var, self.cate_id)#batch_size  * embedding_dim
            cate_seq_embedding = tf.nn.embedding_lookup(cate_id_var, cateid_seq)#batch_size * seq_len * embedding_dim
            if self.use_aux_loss:
                click_neg_seq_embedding = tf.nn.embedding_lookup(item_id_var, click_neg_seq)#batch_size * seq_len * embedding_dim
                cate_neg_seq_embedding = tf.nn.embedding_lookup(cate_id_var, cateid_neg_seq)#batch_size * seq_len * embedding_dim
            # week embedding
            week_var = tf.get_variable("week_var", [8, self.embedding_dim])
            week_embedding = tf.nn.embedding_lookup(week_var, self.week)#batch_size  * embedding_dim
            week_seq_embedding = tf.nn.embedding_lookup(week_var, weeks)#batch_size * seq_len * embedding_dim
            # month embedding
            month_var = tf.get_variable("month_var", [13, self.embedding_dim])
            month_embedding = tf.nn.embedding_lookup(month_var, self.month)#batch_size  * embedding_dim
            month_seq_embedding = tf.nn.embedding_lookup(month_var, months)#batch_size * seq_len * embedding_dim
            # month_diff embedding
            diff_months_var = tf.get_variable("diff_months_var", [12*22, self.embedding_dim])
            diff_months_seq_embedding = tf.nn.embedding_lookup(diff_months_var, diff_months)#batch_size * seq_len * embedding_dim
            # month_diff embedding
            diff_days_var = tf.get_variable("diff_days_var", [12*22*30, self.embedding_dim])
            diff_days_seq_embedding = tf.nn.embedding_lookup(diff_days_var, diff_days)#batch_size * seq_len * embedding_dim
            # month_diff embedding
            convert_mask_var = tf.get_variable("convert_mask_var", [2, self.embedding_dim])
            convert_mask_embedding = tf.nn.embedding_lookup(convert_mask_var, tf.cast(convert_mask,tf.int32))#batch_size * seq_len * embedding_dim
            seq_embedding_sum = tf.reduce_sum(click_seq_embedding, 1)#batch_size  * embedding_dim
            cate_embedding_sum = tf.reduce_sum(cate_seq_embedding, 1)#batch_size  * embedding_dim
            seq_embedding_sum = tf.concat([seq_embedding_sum, cate_embedding_sum], -1)
            item_embedding = tf.concat([item_id_embedding, cate_id_embedding], -1)
            
        #选择模型+_______________
        #——————————————————MLP/LR/wide&deep————————————————————————
        if self.model_type == "lr" or self.model_type == "wd":
            features = [user_id_embedding, item_embedding, week_embedding]
        elif self.model_type == "pnn":
            features = [user_id_embedding, item_embedding, week_embedding, seq_embedding_sum, item_embedding*seq_embedding_sum]
        #—————————————din————————————————————————       
        elif self.model_type == "din":
            with tf.variable_scope('click_din_attention'):
                click_seq_embedding = tf.concat([click_seq_embedding, cate_seq_embedding], -1)
                item_embedding = tf.concat([item_id_embedding, cate_id_embedding], -1)
                click_seq_mask = tf.not_equal(weeks, 0) # 
                click_din = din(tf.expand_dims(item_embedding, 1), click_seq_embedding, click_seq_mask)#(batch_size, embedding_size)
            features = [user_id_embedding, item_embedding, click_din, week_embedding, seq_embedding_sum]
        
        #——————————————dien——————————————————————————
        elif self.model_type == "dien":
            seq_len = tf.ones_like(item_id_embedding[:,0]) * self.seq_len
            with tf.name_scope('rnn_1'):
                click_seq_mask = tf.not_equal(weeks, 0)
                click_seq_embedding = tf.concat([click_seq_embedding, cate_seq_embedding], -1)
                item_embedding = tf.concat([item_id_embedding, cate_id_embedding], -1)
                click_seq_embedding = tf.layers.dropout(click_seq_embedding, self.dropout_rate, training=self.is_train)
                cell = tf.nn.rnn_cell.GRUCell(36)
                rnn_outputs, _ = dynamic_rnn(cell=cell, inputs=click_seq_embedding,
                                             sequence_length=seq_len, dtype=tf.float32,
                                             scope="gru1")
                if self.use_aux_loss:
                    click_neg_seq_embedding = tf.concat([click_neg_seq_embedding, cate_neg_seq_embedding], -1)
                    aux_loss = self.auxiliary_loss(rnn_outputs[:, :-1, :], click_seq_embedding[:, 1:, :],
                                         click_neg_seq_embedding[:, 1:, :],
                                         click_seq_mask[:, 1:], stag="gru")
            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                att_outputs, alphas = din_fcn_attention(item_embedding, rnn_outputs, 36, click_seq_mask,
                                                        softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            with tf.name_scope('rnn_2'):
                rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(36), inputs=rnn_outputs,
                                                         att_scores=tf.expand_dims(alphas, -1),
                                                         sequence_length=seq_len, dtype=tf.float32,
                                                         scope="gru2")
            features = [user_id_embedding, item_embedding, final_state2, week_embedding, seq_embedding_sum]
        elif self.model_type == "dmin":
            click_seq_mask = tf.not_equal(weeks, 0)
            click_seq_embedding = tf.concat([click_seq_embedding, cate_seq_embedding], -1)
            seqs = tf.layers.dropout(click_seq_embedding, self.dropout_rate, training=self.is_train)
            src_mask_click = tf.math.equal(click_seq_mask, False)#batch_size * (seq_len ) #binary
            seqs = multihead_attention(seqs, seqs, seqs, src_mask_click, num_heads=8, 
                                       dropout_rate=self.dropout_rate, training=self.is_train)
            click_interests = ff(seqs, [128, self.embedding_dim*2])#batch_size  * seq_len * model_dim
            if self.use_aux_loss:
                    click_neg_seq_embedding = tf.concat([click_neg_seq_embedding, cate_neg_seq_embedding], -1)
                    aux_loss = self.auxiliary_loss(click_interests[:, :-1, :], click_seq_embedding[:, 1:, :],
                                         click_neg_seq_embedding[:, 1:, :],
                                         click_seq_mask[:, 1:], stag="gru")
            click_interests = multihead_attention(click_interests, click_interests, click_interests, \
                                                  src_mask_click, num_heads=8,
                                                  dropout_rate=self.dropout_rate, training=self.is_train)
            click_interests = ff(seqs, [128, self.embedding_dim*2])#batch_size  * seq_len * model_dim
            item_embedding = tf.concat([item_id_embedding, cate_id_embedding], -1)
            click_din = din(tf.expand_dims(item_embedding, 1), click_interests, click_seq_mask)#(batch_size, embedding_size)
            
            features = [user_id_embedding, item_embedding, click_din, week_embedding, seq_embedding_sum]
        elif self.model_type == "mian":
            with tf.variable_scope("local_att"):
                click_seq_mask = tf.not_equal(weeks, 0)
                seqs = tf.layers.dropout(click_seq_embedding, self.dropout_rate, training=self.is_train)
                src_mask_click = tf.math.equal(click_seq_mask, False)#batch_size * (seq_len ) #binary
                seqs = multihead_attention(seqs, seqs, seqs, src_mask_click, num_heads=8, 
                                           dropout_rate=self.dropout_rate, training=self.is_train)
                click_interests = ff(seqs, [128, self.embedding_dim])#batch_size  * seq_len * model_dim
                item_embedding = item_id_embedding
                out1 = mian_att_unit(tf.expand_dims(item_embedding, 1), click_interests)
                out1 = tf.expand_dims(out1, 1)
                out2 = tf.expand_dims(user_id_embedding, 1)
                inp3 = tf.concat([tf.expand_dims(week_embedding, 1), tf.expand_dims(cate_id_embedding, 1)], 1)
                out3 = mian_att_unit(tf.expand_dims(item_embedding, 1), inp3)
                out3 = tf.expand_dims(out3, 1)
            with tf.variable_scope("global_att"):
                global_out = tf.concat([out1, out2, out3], 1)
                out = mian_att_unit(tf.expand_dims(item_embedding, 1), global_out)
            features = [user_id_embedding, item_embedding, week_embedding, seq_embedding_sum, out]
        #——————————————cmin——————————————————————————
        elif self.model_type == "cmin":
            with tf.variable_scope("prepare_layer"):
                context_absolute_his_embedding = tf.concat([week_seq_embedding], 
                                                           -1)#batch_size*seq_len*(embedding_dim*2)
                context_relative_his_embedding = tf.concat([convert_mask_embedding,diff_months_seq_embedding,
                                                            diff_days_seq_embedding], -1)#batch_size * seq_len * (embedding_dim*3)
                context_absolute_item_embedding = tf.concat([week_embedding], 
                                                  -1)#batch_size * (embedding_dim*2)
                item_embedding = tf.concat([item_id_embedding, cate_id_embedding], -1) #batch_size * 2embedding_dim
                item_his_embedding = tf.concat([click_seq_embedding, cate_seq_embedding,
                                                context_absolute_his_embedding, 
                                                context_relative_his_embedding], 2)#batch_size  * seq_len * (6*embedding_dim)
                #no context1
                #item_his_embedding = tf.concat([click_seq_embedding, cate_seq_embedding], 2)#batch_size  * seq_len * (2*embedding_dim)
                
                self.mask_click = tf.cast(tf.math.not_equal(weeks, 0), tf.float32)
                item_his_embedding_sum = tf.reduce_sum(item_his_embedding, 1) #batch_size  * (6*embedding_dim)

            with tf.variable_scope("click_sequence_interset_extractor"):
                #添加一个特征映射层
                item_model_his_embedding = tf.layers.dense(item_his_embedding, self.model_dim, activation=None)    

                seqs = tf.layers.dropout(item_model_his_embedding, self.dropout_rate, training=self.is_train)
                src_mask_click = tf.math.equal(self.mask_click, 0)#batch_size * (seq_len ) #binary
                seqs = multihead_attention(seqs, seqs, seqs, src_mask_click, num_heads=8, 
                                           dropout_rate=self.dropout_rate, training=self.is_train)
                click_interests = ff(seqs, [128, self.model_dim])#batch_size  * seq_len * model_dim
                click_interests = click_interests * tf.expand_dims(self.mask_click, -1)#batch_size  * seq_len * 1

            with tf.variable_scope("click_sequence_interset_selector"):    
                #添加一个特征映射层
                item_model_embedding = tf.layers.dense(item_embedding, self.model_dim, activation=None)
                
                item_model_embedding = tf.expand_dims(
                    tf.concat([item_model_embedding, context_absolute_item_embedding],-1),1)#B*1*(embedding_dim+model_dim)
                click_interests = tf.concat([click_interests,context_absolute_his_embedding],-1)#B*N*(embedding_dim+model_dim)
                
                ###no context2
                #item_model_embedding = tf.expand_dims(
                #    tf.concat([item_model_embedding],-1),1)#B*1*(embedding_dim+model_dim)
                #click_interests = tf.concat([click_interests],-1)#B*N*(embedding_dim+model_dim)
                
                click_interests = tf.concat([click_interests,item_model_embedding],1)
                click_interests = tf.layers.dense(click_interests, self.model_dim, activation=None)#B*(seq_len+1)*model_dim
                with tf.variable_scope("not_convert_sub_seq_selector"):  
                    not_convert_sub_mask = self.mask_click
                    #not_convert_sub_mask = self.mask_click - self.convert_mask
                    target_mask = tf.expand_dims(tf.ones_like(not_convert_sub_mask[:, 0]), 1)
                    not_convert_sub_mask = tf.concat([not_convert_sub_mask,target_mask],-1)#B*(seq_len+1)
                    src_not_convert_sub_mask = tf.math.equal(not_convert_sub_mask, 0)#B*(seq_len+1)
                    not_convert_interests = multihead_attention(click_interests, click_interests, click_interests, src_not_convert_sub_mask, num_heads=8, dropout_rate=self.dropout_rate, training=self.is_train)
                    not_convert_interests = ff(not_convert_interests, [128, self.model_dim])#batch_size  * seq_len * model_dim
                    not_convert_interests = not_convert_interests * tf.expand_dims(not_convert_sub_mask, -1)
                    not_convert_interest = not_convert_interests[:,-1,:]#batch_size * model_dim
                    not_convert_interests_sum = tf.reduce_sum(not_convert_interests[:,:-1,:], 1)#batch_size * model_dim
                    #not_convert_interests_avg = not_convert_interests_sum/tf.reduce_sum(not_convert_sub_mask,-1,keepdims=True)
                with tf.variable_scope("convert_sub_seq_selector"):  
                    convert_sub_mask = tf.concat([convert_mask, target_mask],-1)#B*(seq_len+1)
                    src_convert_sub_mask = tf.math.equal(convert_sub_mask, 0)#B*seq_len
                    convert_interests = multihead_attention(click_interests, click_interests, click_interests, src_convert_sub_mask, num_heads=8, dropout_rate=self.dropout_rate, training=self.is_train)
                    convert_interests = ff(convert_interests, [128, self.model_dim])#batch_size  * seq_len * model_dim
                    convert_interests = convert_interests * tf.expand_dims(convert_sub_mask, -1)
                    convert_interest = convert_interests[:,-1,:]#batch_size * model_dim
                    convert_interests_sum = tf.reduce_sum(convert_interests[:,:-1,:], 1)#batch_size * model_dim
                    #convert_interests_avg = convert_interests_sum / tf.reduce_sum(convert_sub_mask,-1,keepdims=True)
            features = [user_id_embedding, item_embedding,week_embedding,seq_embedding_sum,
                        item_his_embedding_sum,
                        not_convert_interest,not_convert_interests_sum,
                        convert_interest,convert_interests_sum]
        else:
            features = []
            
        #——————————————————————————————————————————————————————
        with tf.variable_scope("C-Net"):
            inp = tf.concat(features, -1)
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)#
            if self.model_type == "wd":
                dnn1 = tf.layers.dense(bn1, self.dnn_num1, activation=tf.nn.relu, name='f1', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
                dnn1 = tf.layers.dropout(dnn1, self.dropout_rate, training=self.is_train)
                dnn2 = tf.layers.dense(dnn1, self.dnn_num2, activation=tf.nn.relu, name='f2', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
                dnn2 = tf.layers.dropout(dnn2, self.dropout_rate, training=self.is_train)
                dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
                wide = tf.layers.dense(inp, 2, activation=None, name='w1')
                self.y_hat = tf.nn.softmax(dnn3 + wide)
            elif self.model_type == "lr":
                wide = tf.layers.dense(inp, 2, activation=None, name='w1')
                self.y_hat = tf.nn.softmax(wide)
            else:
                dnn1 = tf.layers.dense(bn1, self.dnn_num1, activation=tf.nn.relu, name='f1', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
                dnn1 = tf.layers.dropout(dnn1, self.dropout_rate, training=self.is_train)
                dnn2 = tf.layers.dense(dnn1, self.dnn_num2, activation=tf.nn.relu, name='f2', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
                dnn2 = tf.layers.dropout(dnn2, self.dropout_rate, training=self.is_train)
                dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
                self.y_hat = tf.nn.softmax(dnn3) + 0.00000001
            #转换为batch_size * 2
            ctr_label = tf.concat([tf.reshape(self.label, [-1,1]), tf.reshape(1-self.label, [-1,1])], -1)
            self.target_ph_ = label_smoothing(ctr_label)
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            l2_loss = tf.losses.get_regularization_loss()
            ctr_loss = - tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-8, 1.0)) * self.target_ph_)
            self.loss = ctr_loss + l2_loss
            if self.use_aux_loss and (self.model_type == "dien" or self.model_type == "dmin"):
                self.loss += aux_loss
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(self.loss)
            tf.summary.scalar('loss', self.loss)
            self.t = tf.summary.merge_all()
            
            
    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def evaluate(self, sess, inps,  is_train = False):
        feed_dict = {
            self.label: inps["label"],
            self.user_id: inps["user_id"],
            self.item_id: inps["item_id"],
            self.cate_id: inps["cate_id"],
            self.week: inps["week"],
            self.month: inps["month"],
            self.click_seq: inps["click_seq"],
            self.cateid_seq: inps["cateid_seq"],
            self.convert_mask: inps["convert_mask"],
            self.weeks: inps["weeks"],
            self.months: inps["months"],
            self.diff_months: inps["diff_months"],
            self.diff_days: inps["diff_days"],
            self.click_neg_seq: inps["click_neg_seq"],
            self.cateid_neg_seq: inps["cateid_neg_seq"],
            self.is_train:is_train
        }
        probs, loss = sess.run([self.y_hat, self.loss], feed_dict=feed_dict)
        return probs, loss

    def train(self, sess, inps, is_train = True):
        feed_dict = {
            self.label: inps["label"],
            self.user_id: inps["user_id"],
            self.item_id: inps["item_id"],
            self.cate_id: inps["cate_id"],
            self.week: inps["week"],
            self.month: inps["month"],
            self.click_seq: inps["click_seq"],
            self.cateid_seq: inps["cateid_seq"],
            self.convert_mask: inps["convert_mask"],
            self.weeks: inps["weeks"],
            self.months: inps["months"],
            self.diff_months: inps["diff_months"],
            self.diff_days: inps["diff_days"],
            self.click_neg_seq: inps["click_neg_seq"],
            self.cateid_neg_seq: inps["cateid_neg_seq"],
            self.is_train:is_train
        }
        loss, _, _,t = sess.run([self.loss, self.optimizer, self.update_ops, self.t], feed_dict=feed_dict)
        return loss,t

    def save(self, sess, path, steps):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path, global_step=steps)

        

        
        
        
        
        
        
        
        
        
        
        
        
        