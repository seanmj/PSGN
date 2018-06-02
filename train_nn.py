import sys
import tensorflow as tf
import numpy as np
import cv2
import tflearn
import random
import math
import os
#os.system("chmod +w /unsullied/sharefs/wangmengdi/wangmengdi")
import time
import zlib
import socket
import threading
import Queue
import sys
import tf_nndistance
import cPickle as pickle

from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from template_ffd.model import get_builder
from shapenet.core import cat_desc_to_id
from template_ffd.data.ids import get_example_ids
from template_ffd.metrics.np_impl import np_metrics
from template_ffd.metrics.tf_impl import tf_metrics

from BatchFetcher import *


lastbatch=None
lastconsumed=FETCH_BATCH_SIZE
BATCH_SIZE=32


#def fetch_batch():
#    global lastbatch,lastconsumed
#    if lastbatch is None or lastconsumed+BATCH_SIZE>FETCH_BATCH_SIZE:
#        lastbatch=fetchworker.fetch()
#        lastconsumed=0
#    ret=[i[lastconsumed:lastconsumed+BATCH_SIZE] for i in lastbatch]
#    lastconsumed+=BATCH_SIZE
#    return ret
#def stop_fetcher():
#    fetchworker.shutdown()

def build_graph(resourceid,mode):
    with tf.device('/gpu:%d'%resourceid):
        tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=0.9,soft_placement=True)
#        img_inp=tf.placeholder(tf.float32,shape=(BATCH_SIZE,HEIGHT,WIDTH,3),name='img_inp')
#        pt_gt=tf.placeholder(tf.float32,shape=(BATCH_SIZE,POINTCLOUDSIZE,3),name='pt_gt')
        
        builder = get_builder('plane')
        if(mode=="train"):
            feature,pointcloud = builder.get_train_inputs()
        elif(mode=="test"):
            feature,pointcloud = builder.get_predict_inputs()
        img_inp = feature['image']
        pt_gt = pointcloud
        training_flag = tf.placeholder(tf.bool,name='training_flag')
        x=img_inp
		
        #origin hourglass
        
#192 256
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x0=x
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#96 128
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x1=x
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#48 64
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#24 32
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#12 16
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#6 8
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
        x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
        x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x  
        x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d_transpose(x,32,[5,5],[48,64],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
        x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x2))
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d_transpose(x,16,[5,5],[96,128],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#96 128
        x1=tflearn.layers.conv.conv_2d(x1,16,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x1))
        x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
        x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x2))
        x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x
        x=tflearn.layers.conv.conv_2d(x,128,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x
        x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
        x_additional=tflearn.layers.core.fully_connected(x_additional,2048,activation='linear',weight_decay=1e-4,regularizer='L2')
        x_additional=tf.nn.relu(tf.add(x_additional,tflearn.layers.core.fully_connected(x,2048,activation='linear',weight_decay=1e-3,regularizer='L2')))
        x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
        x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x5))
        x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x5=x  
        x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
        x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x4))
        x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x4=x
        x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
        x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.nn.relu(tf.add(x,x3))
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

#        if(mode=="test"):
#            print "test"
#            BATCH_SIZE=1
        
        x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
        x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
        x_additional=tf.reshape(x_additional,(-1,256,3))
        x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
        x=tf.reshape(x,(-1,32*24,3))
        x=tf.concat([x_additional,x],1)
        x=tf.reshape(x,(-1,OUTPUTPOINTS,3))
        
        
        
        
        
        dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pt_gt,x)#forward-16384*32,backward-1024*32
        mindist=dists_forward
        dist0=mindist[0,:]
        dists_forward=tf.reduce_mean(dists_forward)
        dists_backward=tf.reduce_mean(dists_backward)
        if(mode=="train"):
            loss_nodecay=(dists_forward+dists_backward/2.0)
            loss=loss_nodecay+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1
        if(mode=="test"):
#            loss_nodecay=np_metrics.chamfer(pt_gt, x)
            loss_nodecay=dists_forward+dists_backward
            loss=loss_nodecay

        batchno = tf.Variable(0, dtype=tf.int32)
        optimizer = tf.train.AdamOptimizer(3e-5).minimize(loss,global_step=batchno)
        batchnoinc=batchno.assign(batchno+1)
    return img_inp,x,pt_gt,training_flag,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0



def main(resourceid,keyname):
    if not os.path.exists(dumpdir):
        os.system("mkdir -p %s"%dumpdir)
    img_inp,x,pt_gt,training_flag,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid,"train")
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    #config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True,gpu_options = gpu_options)    
        
    saver=tf.train.Saver()
    with tf.Session(config=config) as sess,\
                open('%s/%s.log'%(dumpdir,keyname),'a') as fout:
        ckpt = tf.train.get_checkpoint_state('./dump')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        trainloss_accs=[0,0,0]
        trainloss_acc0=1e-9
        validloss_accs=[0,0,0]
        validloss_acc0=1e-9
        lastsave=time.time()
        bno=sess.run(batchno)
#        fetchworker.bno=bno//(FETCH_BATCH_SIZE/BATCH_SIZE)
#        fetchworker.start()
        #
        while bno<100000:
            t0=time.time()
            #data,ptcloud,validating=fetch_batch()
            t1=time.time()
            validating=False
            #validating=validating[0]!=0
            if not validating:
#                _,pred,total_loss,trainloss,trainloss1,trainloss2,distmap_0=sess.run([optimizer,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
#                    feed_dict={img_inp:data,pt_gt:ptcloud,training_flag:True})
                _,pred,total_loss,trainloss,trainloss1,trainloss2,distmap_0=sess.run([optimizer,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
                    feed_dict={training_flag:True})
#                trainloss_accs[0]=trainloss_accs[0]*0.99+trainloss
#                trainloss_accs[1]=trainloss_accs[1]*0.99+trainloss1
#                trainloss_accs[2]=trainloss_accs[2]*0.99+trainloss2
#                trainloss_acc0=trainloss_acc0*0.99+1
                #tloss=trainloss1+trainloss2
            else:
                _,pred,total_loss,validloss,validloss1,validloss2,distmap_0=sess.run([batchnoinc,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
                    feed_dict={img_inp:data,pt_gt:ptcloud,training_flag:False})
                validloss_accs[0]=validloss_accs[0]*0.997+validloss
                validloss_accs[1]=validloss_accs[1]*0.997+validloss1
                validloss_accs[2]=validloss_accs[2]*0.997+validloss2
                validloss_acc0=validloss_acc0*0.997+1
                #vloss=validloss1+validloss2
            t2=time.time()
            down=2

            bno=sess.run(batchno)
            if not validating:
                showloss=trainloss
                showloss1=trainloss1
                showloss2=trainloss2
            else:
                showloss=validloss
                showloss1=validloss1
                showloss2=validloss2
            #print >>fout,bno,trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,showloss,showloss1,showloss2,validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss

            
            if bno%10==0:
#                print >>fout,bno,trainloss_accs[1]/trainloss_acc0+trainloss_accs[2]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,validloss_accs[1]/validloss_acc0+validloss_accs[2]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss,time.time()-t0
                print >>fout,bno,trainloss1+trainloss2,time.time()-t0
            if bno%128==0:
                fout.flush()
            if time.time()-lastsave>900:
                saver.save(sess,'%s/'%dumpdir+keyname+".ckpt")
                lastsave=time.time()
            #print 'bno',bno,'trainloss',trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,'validloss',validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss,t1-t0,t2-t1,time.time()-t0,fetchworker.queue.qsize()
            if bno%10==0:
#                print 'bno',bno,'trainloss',trainloss_accs[1]/trainloss_acc0+trainloss_accs[2]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,'validloss',validloss_accs[1]/validloss_acc0+validloss_accs[2]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,time.time()-t0
                print 'bno',bno,'trainloss',trainloss1+trainloss2,time.time()-t0
        saver.save(sess,'%s/'%dumpdir+keyname+".ckpt") 
        print keyname+'.ckpt has been saved'

# def dumppredictions(resourceid,keyname,valnum):
    # img_inp,x,pt_gt,training_flag,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid,"test")
    # config=tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # config.allow_soft_placement=True
    # saver=tf.train.Saver()
    # fout = open("%s/%s.v.pkl"%(dumpdir,keyname),'wb')
    # with tf.Session(config=config) as sess:
        # #sess.run(tf.initialize_all_variables())
        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))
        # fetchworker.bno=0
        # fetchworker.start()
        # cnt=0
        # #
        # for i in xrange(0,300000):
            # t0=time.time()
            # data,ptcloud,validating=fetch_batch()
            # validating=validating[0]!=0
            # if not validating:
                # continue
            # cnt+=1
            # pred,distmap=sess.run([x,mindist],feed_dict={img_inp:data,pt_gt:ptcloud,training_flag:False})
            # pickle.dump((i,data,ptcloud,pred,distmap),fout,protocol=-1)
            # print i,'time',time.time()-t0,cnt
            # if cnt>=valnum:
                # break
    # fout.close()

def test(resourceid,keyname):
    img_inp,x,pt_gt,training_flag,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid,"test")
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    saver=tf.train.Saver()
    cat_id=cat_desc_to_id('plane')
    length=len(get_example_ids(cat_id,'eval'))
    fout = open("%s/%s.v.txt"%(dumpdir,keyname),'wb')
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))
        values=[]
        for i in xrange(0,length):
            _,pred,total_loss,_,forward,backward,_=sess.run([batchnoinc,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
                    feed_dict={training_flag:False})
            
            print i,total_loss
            values.append(total_loss)
        print np.mean(values)
        
        

if __name__=='__main__':
    resourceid = 0
#    datadir,dumpdir,cmd,valnum="data","dump","predict",3
    for pt in sys.argv[1:]:
#        if pt[:5]=="data=":
#            datadir = pt[5:]
        if pt[:5]=="dump=":
            dumpdir = pt[5:]
#        elif pt[:4]=="num=":
#            valnum = int(pt[4:])
        else:
            cmd = pt
#    if datadir[-1]=='/':
#        datadir = datadir[:-1]
    if dumpdir[-1]=='/':
        dumpdir = dumpdir[:-1]
#    assert os.path.exists(datadir),"data dir not exists"
    os.system("mkdir -p %s"%dumpdir)
#    fetchworker=BatchFetcher(datadir)
#    print "datadir=%s dumpdir=%s num=%d cmd=%s started"%(datadir,dumpdir,valnum,cmd)
    
    keyname=os.path.basename(__file__).rstrip('.py')
#    try:
    if cmd=="train":
        main(resourceid,keyname)
    elif cmd=="test":
            #dumppredictions(resourceid,keyname,valnum)
        test(resourceid,keyname)
    else:
        assert False,"format wrong"
#    finally:
#        stop_fetcher()
