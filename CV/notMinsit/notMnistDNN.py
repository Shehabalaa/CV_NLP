import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle


pickle_file='notMNIST.pickle'

with open(pickle_file,'rb') as f:
    save=pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print("Trainging set :",train_dataset.shape,train_labels.shape)
    print("Validating set :",valid_dataset.shape,valid_labels.shape)
    print("Testing set :",test_dataset.shape,train_labels.shape)


#reformating data
img_size=28
labels_size=10

def reformat(data,labels):
    data = data.reshape((-1,img_size*img_size)).astype(np.float32)
    labels = (np.arange(labels_size)==labels[:,None]).astype(np.float32)
    return data,labels
    
train_dataset,train_labels = reformat(train_dataset,train_labels)
valid_dataset,valid_labels = reformat(valid_dataset,valid_labels)
test_dataset,test_labels = reformat(test_dataset,test_labels)

print("\n\nTrainging set :",train_dataset.shape,train_labels.shape)
print("Validating set :",valid_dataset.shape,valid_labels.shape)
print("Testing set :",test_dataset.shape,train_labels.shape)




#setting up tensorflow
#firstly with GD then with SGD
#1.GD
''' 
graph=tf.Graph()
with graph.as_default():
    #define constants
    tf_train_dataset=tf.constant(train_dataset[:train_subset,:])
    tf_train_labels=tf.constant(train_labels[:train_subset])
    tf_test_dataset=tf.constant(test_dataset[:train_subset,:])
    tf_valid_dataset=tf.constant(valid_dataset[:train_subset,:])
    
    #define variables
    weights = tf.Variable(tf.truncated_normal([img_size*img_size,labels_size]))
    biases = tf.Variable(tf.zeros([labels_size]))
    
    #define training
    logits = tf.matmul(tf_train_dataset,weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
    #optimizing
    optimizer=tf.train.GradientDescentOptimizer(.5).minimize(loss)

    #no calc predictions
    train_predictions = tf.nn.softmax(logits)
    valid_predictions = tf.nn.softmax(tf.matmul(tf_valid_dataset,weights) + biases)
    test_predictions = tf.nn.softmax(tf.matmul(tf_test_dataset,weights) + biases)
    
    
#now runc computatino and iterate
num_steps=301

def accuracy(predictions,labels):
    return(100.0*np.sum(np.argmax(predictions,1) == np.argmax(labels,1))/labels.shape[0] )




with tf.Session(graph=graph) as session:
    #first we intialize variables
    tf.global_variables_initializer().run()
    print("Itialization")
    for step in range(num_steps):
        _,l,predictions = session.run([optimizer,loss,train_predictions])
        if(step%100==0):
            print("Loss at step %d: %f"%(step,l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset,:]))
            print("Validation accuracy:%.1f%%"% accuracy(valid_predictions.eval(),valid_labels[:train_subset,:]))
    
    print("Test accuracy:%.1f%%"% accuracy(test_predictions.eval(),test_labels))
  

#-----------------------------------------------------------
'''

#2. SGD
hidden_nodes_1 = 1024
hidden_nodes_2 = int(hidden_nodes_1 * 0.5)
hidden_nodes_3 = int(hidden_nodes_2 * np.power(0.5, 2))
hidden_nodes_4 = int(hidden_nodes_3 * np.power(0.5, 3))
hidden_nodes_5 = int(hidden_nodes_4 * np.power(0.5, 4))
keep_prob=.5
beta=0.001
learning_rate=.5
batch_size=128

with tf.device('/gpu:0'):
    #define constants
    tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,img_size*img_size))
    tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,labels_size))
    tf_test_dataset=tf.constant(test_dataset)
    tf_valid_dataset=tf.constant(valid_dataset)
    
    #define variables
    weights1 = tf.Variable(tf.truncated_normal([img_size*img_size,hidden_nodes_1]))
    biases1 = tf.Variable(tf.zeros([hidden_nodes_1]))
    
    weights2 = tf.Variable(tf.truncated_normal([hidden_nodes_1,hidden_nodes_2]))
    biases2 = tf.Variable(tf.zeros([hidden_nodes_2]))
    
    weights3 = tf.Variable(tf.truncated_normal([hidden_nodes_2,hidden_nodes_3]))
    biases3 = tf.Variable(tf.zeros([hidden_nodes_3]))
    
    weights4 = tf.Variable(tf.truncated_normal([hidden_nodes_3,hidden_nodes_4]))
    biases4 = tf.Variable(tf.zeros([hidden_nodes_4]))
    
    weights5 = tf.Variable(tf.truncated_normal([hidden_nodes_4,hidden_nodes_5]))
    biases5 = tf.Variable(tf.zeros([hidden_nodes_5]))
    
    weights6 = tf.Variable(tf.truncated_normal([hidden_nodes_5,labels_size]))
    biases6 = tf.Variable(tf.zeros([labels_size]))
    #--------------------
    nnParams =[(weights1,biases1),(weights2,biases2),(weights3,biases3),(weights4,biases4),
               (weights5,biases5),(weights6,biases6)]
    #define training
    logits = tf_train_dataset
    count = 1
    for w,b in nnParams:
        logits = tf.matmul(logits,w) + b
        if(count!=len(nnParams)):
            logits = tf.nn.relu(logits)
            logits = tf.nn.dropout(logits,keep_prob)
        count += 1
  
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits)
    loss = tf.reduce_mean(loss+beta*(tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2)+tf.nn.l2_loss(weights3)+tf.nn.l2_loss(weights4)+tf.nn.l2_loss(weights5)+tf.nn.l2_loss(weights6)))
    
    #optimizing
    #global_step = tf.Variable(0)
    #learning_rate = tf.train.exponential_decay(0.5, global_step,100000,0.96,staircase=True)
    optimizer=tf.train.GradientDescentOptimizer(.5).minimize(loss)#,global_step=global_step)

    #no calc predictions
    train_predictions = tf.nn.softmax(logits)
    valid_predictions = tf_valid_dataset
    test_predictions = tf_test_dataset
    count = 1
    for w,b in nnParams:
        valid_predictions = tf.matmul(valid_predictions,w)+b
        test_predictions = tf.matmul(test_predictions,w)+b
        if(count!=len(nnParams)):
            valid_predictions = tf.nn.relu(valid_predictions)
            test_predictions = tf.nn.relu(test_predictions)
        count += 1
        
    valid_predictions = tf.nn.softmax(valid_predictions)
    test_predictions = tf.nn.softmax(test_predictions)
    

 
#now runc computatino and iterate
num_steps=20001

def accuracy(predictions,labels):
    return(100.0*np.sum(np.argmax(predictions,1) == np.argmax(labels,1))/labels.shape[0] )

train_dataset = train_dataset[:10000,:]
train_labels = train_labels[:10000,:]
    
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    #first we intialize variables
    tf.global_variables_initializer().run()
    print("Itialization")
    for step in range(num_steps):  
        offset=step*batch_size % (train_labels.shape[0]-batch_size)
        batch_data=train_dataset[offset : (offset+batch_size),:]
        batch_labels=train_labels[offset : (offset+batch_size),:]
        feed_dict={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        _,l,predictions = session.run([optimizer,loss,train_predictions],feed_dict=feed_dict)
        if(step%500==0):
            print("Minibatch loss at step %d: %f"%(step,l))
            print('Minibatch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print("Minibatch validation accuracy:%.1f%%"% accuracy(valid_predictions.eval(),valid_labels))
    
    print("Test accuracy:%.1f%%"% accuracy(test_predictions.eval(),test_labels))
    



    
    
    
    
    
    