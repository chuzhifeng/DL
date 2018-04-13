import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#this is data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

#hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28 #MNIST data input(img shape:28*28)
n_steps = 28 #time steps
n_hidden_unis = 128 #neuros in hidden layer
n_classes = 10 #MNIST classes (0-9 digits)

#tf Graph input 
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

# Define weights 
weights = {
	#(28,128)
	'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
	#(128,10)
	'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}

biaes = {
	#(128,)
	'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
	#(10,)
	'out':tf.Variable(tf.constant(0.1,shape= [n_classes,]))
}

def RNN(X,weights,biaes):
	#hidden layer for input to cell
	################################
	#X (128 batch,28 steps,28 inputs)
	# ==>(128*28,28 inputs)
	X = tf.reshape(X,[-1,n_inputs])

	#X_in ==> (128 batch*28 steps,128 hidden)
	X_in = tf.matmul(X,weights['in'])+biaes['in']

	#X_in ==> (128 batch,28 steps,128 hidden)
	X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_unis])

	#cell
	################################
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias = 1.0,state_is_tuple = True)
	#lstm cell is divided into two parts(c_state,m_state)
	_init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)

	outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state = _init_state,time_major=False)

	#hidden layer for output as the final results
	#################################
	results = tf.matmul(states[1],weights['out'])+biaes['out']

	# unpack to list[(batch,outputs)..]*step
	# outputs = tf.unpack(tf.transpose(outputs,[1,0,2])) # states is the last outputs
	# results = tf.matmul(outputs[-1],weights['out'])+biaes['out']
	return results


pred = RNN(x,weights,biaes)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_op = tf.tran.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuary = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variable_initializer()	
with tf.Session() as sess:
	sess.run(init)
	step = 0
	while step * batch_size < training_iters:
		batch_xs,batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
		sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys,})
step += 1